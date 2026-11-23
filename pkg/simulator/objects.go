package simulator

import (
	"fmt"
	"math/rand"

	"crv/pkg/schema"
	"crv/pkg/types"
)

// NewObject builds a fully populated object of the given kind/version.
// Every field in the version's schema gets a deterministic-ish random value.
func NewObject(k *schema.Kind, version, namespace, name string, rng *rand.Rand) (*types.Object, error) {
	vs, err := k.Version(version)
	if err != nil {
		return nil, err
	}
	obj := &types.Object{
		Group:   k.Group,
		Kind:    k.Name,
		Version: version,
		Metadata: types.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				"app":  k.Name,
				"seed": fmt.Sprintf("%d", rng.Int63()),
			},
			Annotations: map[string]string{
				"crv.io/created-by": "simulator",
			},
		},
		Spec:   map[string]any{},
		Status: map[string]any{},
	}
	for _, f := range vs.Fields {
		val := valueFor(f.Type, rng)
		putField(obj, f.Path, val)
	}
	return obj, nil
}

func valueFor(t schema.FieldType, rng *rand.Rand) any {
	switch t {
	case schema.TypeString:
		return randomWord(rng)
	case schema.TypeInt:
		return int64(1 + rng.Intn(1000))
	case schema.TypeBool:
		return rng.Intn(2) == 0
	case schema.TypeFloat:
		return rng.Float64() * 100
	case schema.TypeList:
		n := 1 + rng.Intn(3)
		out := make([]any, n)
		for i := range out {
			out[i] = randomWord(rng)
		}
		return out
	case schema.TypeObject:
		return map[string]any{"inner": randomWord(rng)}
	default:
		return nil
	}
}

// MutateValue returns a new value of the same type, used by RMW traffic.
func MutateValue(t schema.FieldType, rng *rand.Rand) any {
	return valueFor(t, rng)
}

func putField(obj *types.Object, path string, val any) {
	segs := splitDotted(path)
	var node map[string]any
	switch segs[0] {
	case "spec":
		node = obj.Spec
	case "status":
		node = obj.Status
	default:
		panic("bad field path " + path)
	}
	for _, seg := range segs[1 : len(segs)-1] {
		next, ok := node[seg].(map[string]any)
		if !ok {
			next = map[string]any{}
			node[seg] = next
		}
		node = next
	}
	node[segs[len(segs)-1]] = val
}

func splitDotted(p string) []string {
	var out []string
	start := 0
	for i := 0; i < len(p); i++ {
		if p[i] == '.' {
			out = append(out, p[start:i])
			start = i + 1
		}
	}
	return append(out, p[start:])
}
