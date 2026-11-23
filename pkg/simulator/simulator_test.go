package simulator

import (
	"encoding/json"
	"strings"
	"testing"

	"crv/pkg/client"
	"crv/pkg/convert"
)

func TestGeneratedCatalogueShape(t *testing.T) {
	reg, err := GenerateRegistry(7)
	if err != nil {
		t.Fatal(err)
	}
	kinds := reg.All()
	if len(kinds) != len(templates) {
		t.Fatalf("kind count = %d, want %d", len(kinds), len(templates))
	}
	if len(kinds) < 30 {
		t.Fatalf("simulation must have dozens of kinds, got %d", len(kinds))
	}
	totalVersions := 0
	for _, k := range kinds {
		if len(k.Versions) < 2 {
			t.Fatalf("kind %s has fewer than 2 versions", k.Name)
		}
		totalVersions += len(k.Versions)
		// every non-last version needs an edge with at least one rule
		for i := 0; i < len(k.Versions)-1; i++ {
			from := k.Versions[i].Name
			if len(k.Rules[from]) == 0 {
				t.Fatalf("kind %s edge %s has no evolution rules", k.Name, from)
			}
		}
		if _, err := k.FieldSet(k.StorageVersion); err != nil {
			t.Fatalf("kind %s bad storage version: %v", k.Name, err)
		}
	}
	t.Logf("generated %d kinds, %d version schemas total (%.1f versions/kind)",
		len(kinds), totalVersions, float64(totalVersions)/float64(len(kinds)))
}

func TestEveryKindEveryVersionRoundTrip(t *testing.T) {
	p, err := NewPlatform(11)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stop()
	for _, k := range p.Registry.All() {
		latest := k.Versions[len(k.Versions)-1].Name
		obj, err := NewObject(k, latest, "ns", "probe", p.Rand)
		if err != nil {
			t.Fatalf("build %s: %v", k.Name, err)
		}
		creator := client.New(p.Store, k.Name, latest, client.Preserving)
		if _, err := creator.Create(obj); err != nil {
			t.Fatalf("create %s: %v", k.Name, err)
		}
		// read it back through EVERY served version, then again in the
		// latest version: nothing may be lost across the whole chain.
		for _, vs := range k.Versions {
			if _, err := p.Store.Get(k.Name, "ns", "probe", vs.Name); err != nil {
				t.Fatalf("read %s as %s: %v", k.Name, vs.Name, err)
			}
		}
		again, err := p.Store.Get(k.Name, "ns", "probe", latest)
		if err != nil {
			t.Fatal(err)
		}
		if !mapsEqualJSON(obj.Spec, again.Spec) {
			t.Fatalf("%s latest-view drift after fanning out to all versions\n got %#v\nwant %#v",
				k.Name, again.Spec, obj.Spec)
		}
	}
}

func TestSeededStoreNormalizesToStorageVersion(t *testing.T) {
	p, err := NewPlatform(13)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stop()
	if err := p.SeedObjects(4); err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, k := range p.Registry.All() {
		counts := p.Store.CountByStorageVersion(k.Name)
		for v, n := range counts {
			if v != k.StorageVersion {
				t.Fatalf("kind %s has %d objects in non-storage version %s", k.Name, n, v)
			}
			total += n
		}
	}
	if total != len(p.Registry.All())*4 {
		t.Fatalf("stored object count = %d", total)
	}
}

func TestMixedTrafficLosesNoFields(t *testing.T) {
	p, err := NewPlatform(17)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stop()
	if err := p.SeedObjects(3); err != nil {
		t.Fatal(err)
	}
	stats := p.RunTraffic(2000, 8)
	if stats.InvariantFails.Load() != 0 {
		t.Fatalf("field-loss invariant violated %d times: %s",
			stats.InvariantFails.Load(), stats.FirstViolation)
	}
	t.Logf("traffic: %d ops (reads=%d rmw=%d creates=%d), success=%d conflicts=%d faultRejects=%d",
		stats.Iterations.Load(), stats.Reads.Load(), stats.ReadModifyWrite.Load(), stats.Creates.Load(),
		stats.Success.Load(), stats.Conflicts.Load(), stats.RejectedFaults.Load())
}

func TestTrafficUnderConversionFaults(t *testing.T) {
	p, err := NewPlatform(19)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Stop()
	if err := p.SeedObjects(3); err != nil {
		t.Fatal(err)
	}
	// fault three random kinds; their traffic must fail explicitly while
	// other kinds keep working
	kinds := p.Registry.All()
	p.Service.SetFault(kinds[0].Name, convert.Fault{Down: true})
	p.Service.SetFault(kinds[7].Name, convert.Fault{Down: true})
	p.Service.SetFault(kinds[15].Name, convert.Fault{Down: true})

	stats := p.RunTraffic(1000, 6)
	if stats.InvariantFails.Load() != 0 {
		t.Fatalf("invariant failure during faults: %s", stats.FirstViolation)
	}
	if stats.RejectedFaults.Load() == 0 {
		t.Fatal("expected explicit rejections while conversion service was down")
	}
	// healthy kinds must still succeed
	if stats.Success.Load() == 0 {
		t.Fatal("healthy kinds made no successful progress during partial outage")
	}
	// nothing bad was written: all records still decode in storage shape
	for _, k := range kinds {
		for _, key := range p.Store.KeysForVersion(k.Name, k.StorageVersion) {
			ns, name := splitKey(key)
			if _, raw, ok := p.Store.RawRecord(k.Name, ns, name); !ok || raw == nil {
				t.Fatalf("unreadable record %s/%s", k.Name, key)
			}
		}
	}
	t.Logf("partial outage: rejected=%d success=%d, zero bad records",
		stats.RejectedFaults.Load(), stats.Success.Load())
}

// mapsEqualJSON compares maps through canonical JSON so int64/float64
// number representations do not cause false negatives.
func mapsEqualJSON(a, b map[string]any) bool {
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	return string(ja) == string(jb)
}

func splitKey(key string) (string, string) {
	if i := strings.Index(key, "/"); i >= 0 {
		return key[:i], key[i+1:]
	}
	return "", key
}
