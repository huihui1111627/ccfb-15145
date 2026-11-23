// Package simulator builds a realistic-looking container platform: dozens
// of custom resource kinds (databases, queues, ML jobs, ...), each with
// several schema versions produced by repeated field renames, splits,
// scalar->list changes and additions, plus mixed-version client traffic.
//
// Everything is deterministic from a seed so acceptance runs are
// reproducible.
package simulator

import (
	"fmt"
	"hash/fnv"
	"math/rand"
	"strings"

	"crv/pkg/schema"
)

// kindTemplate describes one simulated resource family.
type kindTemplate struct {
	group        string
	name         string
	versions     int
	baseFields   int
	storageIndex int // which version is the initial storage version
	evolutions   int // mutations per adjacent-version edge
}

// templates covers 36 kinds in 6 resource groups, 2..5 versions each.
var templates = []kindTemplate{
	// databases
	{"data.platform.io", "PostgresInstance", 4, 6, 0, 2},
	{"data.platform.io", "MySQLInstance", 3, 5, 0, 2},
	{"data.platform.io", "RedisCluster", 5, 6, 1, 2},
	{"data.platform.io", "MongoReplicaSet", 3, 5, 0, 1},
	{"data.platform.io", "ClickHouseCluster", 4, 7, 2, 2},
	{"data.platform.io", "CassandraKeyspace", 2, 5, 0, 1},
	// messaging
	{"messaging.platform.io", "KafkaTopic", 4, 6, 0, 2},
	{"messaging.platform.io", "RabbitMQQueue", 3, 5, 0, 2},
	{"messaging.platform.io", "PulsarTenant", 2, 5, 0, 1},
	{"messaging.platform.io", "NATSStream", 3, 5, 1, 2},
	{"messaging.platform.io", "RocketMQGroup", 4, 6, 0, 2},
	{"messaging.platform.io", "RedpandaCluster", 5, 7, 2, 2},
	// machine learning
	{"ml.platform.io", "TrainingJob", 5, 7, 1, 3},
	{"ml.platform.io", "InferenceService", 4, 6, 0, 2},
	{"ml.platform.io", "NotebookSession", 3, 5, 0, 2},
	{"ml.platform.io", "FeatureStore", 2, 5, 0, 1},
	{"ml.platform.io", "PipelineRun", 4, 6, 2, 2},
	{"ml.platform.io", "ModelRegistry", 3, 5, 0, 1},
	// networking / ingress
	{"network.platform.io", "VirtualService", 4, 6, 0, 2},
	{"network.platform.io", "GatewayPolicy", 3, 5, 0, 2},
	{"network.platform.io", "DNSEndpoint", 2, 4, 0, 1},
	{"network.platform.io", "TrafficMirror", 3, 5, 1, 2},
	{"network.platform.io", "ServiceMesh", 4, 6, 0, 2},
	{"network.platform.io", "CertificateBundle", 5, 6, 2, 2},
	// scheduling / compute
	{"compute.platform.io", "GPUQuota", 3, 5, 0, 2},
	{"compute.platform.io", "SpotFleet", 4, 6, 0, 2},
	{"compute.platform.io", "BatchQueue", 2, 5, 0, 1},
	{"compute.platform.io", "AutoscalerPolicy", 4, 6, 1, 2},
	{"compute.platform.io", "NodePool", 5, 7, 2, 3},
	{"compute.platform.io", "Reservation", 3, 5, 0, 1},
	// ops / security
	{"ops.platform.io", "BackupPolicy", 4, 6, 0, 2},
	{"ops.platform.io", "AlertRule", 3, 5, 0, 2},
	{"ops.platform.io", "ScanTarget", 2, 4, 0, 1},
	{"ops.platform.io", "SecretBinding", 3, 5, 1, 2},
	{"ops.platform.io", "AuditSink", 4, 6, 0, 2},
	{"ops.platform.io", "CostBudget", 5, 6, 2, 2},
}

var fieldPool = []string{
	"replicas", "image", "storage_gb", "endpoint", "port", "host",
	"timeout_sec", "retries", "owner", "tier", "zone", "schedule",
	"cpu_limit", "memory_mb", "gpu_count", "topic_name", "partitions",
	"retention_hours", "model_uri", "batch_size", "learning_rate",
	"max_idle", "backup_window", "encryption_key", "vpc_id", "cidr",
	"priority", "queue_depth", "checkpoint_path", "frame_work",
}

var allTypes = []schema.FieldType{
	schema.TypeString, schema.TypeInt, schema.TypeBool, schema.TypeFloat, schema.TypeList,
}

// field is one simulated field path with its value type in a given version.
type field struct {
	path string
	typ  schema.FieldType
}

// GenerateRegistry builds the full simulated resource catalogue.
func GenerateRegistry(seed int64) (*schema.Registry, error) {
	reg := schema.NewRegistry()
	for _, t := range templates {
		k, err := generateKind(t, seed)
		if err != nil {
			return nil, err
		}
		if err := reg.Register(k); err != nil {
			return nil, err
		}
	}
	return reg, nil
}

func generateKind(t kindTemplate, seed int64) (*schema.Kind, error) {
	rng := rand.New(rand.NewSource(seed + hashSeed(t.group+t.name)))

	// base fields for v1
	used := map[string]bool{}
	current := []field{}
	shuffled := rng.Perm(len(fieldPool))
	for i := 0; i < t.baseFields && i < len(shuffled); i++ {
		name := fieldPool[shuffled[i]]
		used[name] = true
		current = append(current, field{"spec." + name, allTypes[rng.Intn(len(allTypes))]})
	}
	// every kind has two stable status fields
	current = append(current,
		field{"status.phase", schema.TypeString},
		field{"status.observed_generation", schema.TypeInt},
	)

	versions := make([]*schema.VersionSchema, 0, t.versions)
	rules := map[string][]*schema.ConversionRule{}
	for vi := 0; vi < t.versions; vi++ {
		vs := &schema.VersionSchema{Name: fmt.Sprintf("v%d", vi+1)}
		for _, f := range current {
			vs.Fields = append(vs.Fields, schema.FieldDef{Path: f.path, Type: f.typ})
		}
		if vi == t.versions-1 {
			vs.Deprecated = false
		}
		versions = append(versions, vs)

		if vi == t.versions-1 {
			break
		}
		edgeRules, next := evolve(current, used, t.evolutions, rng)
		rules[vs.Name] = edgeRules
		current = next
	}

	storageIndex := t.storageIndex
	if storageIndex >= t.versions {
		storageIndex = 0
	}
	return &schema.Kind{
		Group:          t.group,
		Name:           t.name,
		Versions:       versions,
		StorageVersion: versions[storageIndex].Name,
		Rules:          rules,
	}, nil
}

// evolve applies a few schema mutations and returns the edge rules plus the
// next version's field list.
func evolve(current []field, used map[string]bool, n int, rng *rand.Rand) ([]*schema.ConversionRule, []field) {
	next := append([]field(nil), current...)
	var rules []*schema.ConversionRule

	// candidate pool: mutable spec fields only (status stays stable)
	candidates := make([]int, 0, len(next))
	for i, f := range next {
		if isSpec(f.path) {
			candidates = append(candidates, i)
		}
	}
	rng.Shuffle(len(candidates), func(i, j int) {
		candidates[i], candidates[j] = candidates[j], candidates[i]
	})

	mutations := 0
	for _, idx := range candidates {
		if mutations >= n {
			break
		}
		f := next[idx]
		choice := rng.Intn(4)
		switch choice {
		case 0: // rename the leaf segment, preserving the full path
			leaf := leafName(f.path)
			newLeaf := leaf + "_renamed"
			newPath := parentPath(f.path) + newLeaf
			if used[newPath] {
				continue
			}
			used[newPath] = true
			rules = append(rules, &schema.ConversionRule{
				Kind: schema.RuleRename, Source: f.path, Sink: newPath,
			})
			next[idx] = field{newPath, f.typ}
			mutations++
		case 1: // add a brand-new field
			name := pickUnused(used, rng)
			if name == "" {
				continue
			}
			typ := allTypes[rng.Intn(len(allTypes))]
			rules = append(rules, &schema.ConversionRule{
				Kind: schema.RuleAdd, Sink: "spec." + name, Default: defaultValue(typ, rng),
			})
			next = append(next, field{"spec." + name, typ})
			mutations++
		case 2: // split one string field into two string fields
			//         (host-ish -> host + port-ish), merged back with a
			//         separator on downgrade so the edge is fully reversible.
			if f.typ != schema.TypeString {
				continue
			}
			base := leafName(f.path)
			// never re-split a field produced by an earlier split: chained
			// splits create empty-tail values that are legal but confusing
			if strings.HasSuffix(base, "_host") || strings.HasSuffix(base, "_port") {
				continue
			}
			parent := parentPath(f.path)
			a, b := base+"_host", base+"_port"
			pathA, pathB := parent+a, parent+b
			if used[pathA] || used[pathB] {
				continue
			}
			used[pathA], used[pathB] = true, true
			rules = append(rules, &schema.ConversionRule{
				Kind: schema.RuleSplit,
				Sources: [][2]string{
					{f.path, pathA},
					{f.path, pathB},
				},
				MergeSep: ":",
			})
			// replace the one field with two string fields
			next[idx] = field{pathA, schema.TypeString}
			next = append(next, field{pathB, schema.TypeString})
			mutations++
		case 3: // scalar -> list
			if f.typ == schema.TypeList || f.typ == schema.TypeObject {
				continue
			}
			rules = append(rules, &schema.ConversionRule{
				Kind: schema.RuleWrapList, Source: f.path, Sink: f.path + "_list",
			})
			next[idx] = field{f.path + "_list", schema.TypeList}
			mutations++
		}
	}
	// Guarantee every adjacent-version edge carries at least one rule even
	// when random type choices all skipped. Add a fresh field, synthesizing
	// a unique name if the shared pool is exhausted.
	if len(rules) == 0 {
		name := pickUnused(used, rng)
		if name == "" {
			name = fmt.Sprintf("extra_field_%d", len(used))
			used[name] = true
		}
		typ := allTypes[rng.Intn(len(allTypes))]
		rules = append(rules, &schema.ConversionRule{
			Kind: schema.RuleAdd, Sink: "spec." + name, Default: defaultValue(typ, rng),
		})
		next = append(next, field{"spec." + name, typ})
	}
	return rules, next
}

func pickUnused(used map[string]bool, rng *rand.Rand) string {
	start := rng.Intn(len(fieldPool))
	for i := 0; i < len(fieldPool); i++ {
		name := fieldPool[(start+i)%len(fieldPool)]
		if !used[name] {
			used[name] = true
			return name
		}
	}
	return ""
}

func defaultValue(t schema.FieldType, rng *rand.Rand) any {
	switch t {
	case schema.TypeString:
		return "default-" + randomWord(rng)
	case schema.TypeInt:
		return int64(rng.Intn(100))
	case schema.TypeBool:
		return false
	case schema.TypeFloat:
		return rng.Float64()
	case schema.TypeList:
		return []any{randomWord(rng)}
	default:
		return nil
	}
}

var words = []string{"alpha", "bravo", "cosmo", "delta", "echo", "falcon", "gemini", "helix", "ion", "juno"}

func randomWord(rng *rand.Rand) string {
	return words[rng.Intn(len(words))] + "-" + fmt.Sprintf("%d", rng.Intn(1000))
}

func isSpec(path string) bool {
	return len(path) > 5 && path[:5] == "spec."
}

func leafName(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '.' {
			return path[i+1:]
		}
	}
	return path
}

// parentPath returns the dotted path including its trailing dot, or "" for
// a top-level leaf. Used when renaming while keeping the field's location.
func parentPath(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '.' {
			return path[:i+1]
		}
	}
	return ""
}

func hashSeed(s string) int64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return int64(h.Sum64() % 100000)
}
