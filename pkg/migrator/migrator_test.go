package migrator

import (
	"strings"
	"testing"
	"time"

	"crv/pkg/convert"
	"crv/pkg/schema"
	"crv/pkg/storage"
	"crv/pkg/types"
)

func buildStack(t *testing.T) (*Migrator, *storage.Store, *convert.Service, *schema.Registry) {
	t.Helper()
	reg := schema.NewRegistry()
	k := &schema.Kind{
		Group: "test.io", Name: "Widget", StorageVersion: "v1",
		Versions: []*schema.VersionSchema{
			{Name: "v1", Fields: []schema.FieldDef{
				{Path: "spec.name", Type: schema.TypeString},
				{Path: "spec.count", Type: schema.TypeInt},
			}},
			{Name: "v2", Fields: []schema.FieldDef{
				{Path: "spec.display_name", Type: schema.TypeString},
				{Path: "spec.count", Type: schema.TypeInt},
				{Path: "spec.region", Type: schema.TypeString, Optional: true},
			}},
		},
		Rules: map[string][]*schema.ConversionRule{
			"v1": {
				{Kind: schema.RuleRename, Source: "spec.name", Sink: "spec.display_name"},
				{Kind: schema.RuleAdd, Sink: "spec.region", Default: "cn-east-1"},
			},
		},
	}
	if err := reg.Register(k); err != nil {
		t.Fatal(err)
	}
	svc := convert.NewService(convert.NewEngine(reg))
	st := storage.New(reg, svc)
	// pace migrations so tests can observe partial progress and faults
	mg := New(reg, st).WithStepInterval(25 * time.Millisecond)
	return mg, st, svc, reg
}

func seed(t *testing.T, st *storage.Store, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		obj := &types.Object{
			Group: "test.io", Kind: "Widget", Version: "v1",
			Metadata: types.ObjectMeta{Name: string(rune('a'+i%26)) + itoa(i), Namespace: "ns"},
			Spec:     map[string]any{"name": "n-" + itoa(i), "count": int64(i)},
		}
		if _, err := st.Create(obj); err != nil {
			t.Fatal(err)
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func waitFor(t *testing.T, mg *Migrator, kind string) *Progress {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		p := mg.Progress(kind)
		if p != nil && p.State == StateCompleted {
			return p
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("migration did not complete in time")
	return nil
}

func TestMigrationRewritesAllObjectsAndKeepsSemantics(t *testing.T) {
	mg, st, _, reg := buildStack(t)
	defer mg.Stop()
	seed(t, st, 25)

	// snapshot v1 semantics before migration
	before := map[string]any{}
	objs, _ := st.List("Widget", "v1")
	for _, o := range objs {
		before[o.Metadata.Name] = o.Spec["name"]
	}

	p, err := mg.SwitchStorage("Widget", "v2")
	if err != nil {
		t.Fatal(err)
	}
	if p.FromVersion != "v1" || p.ToVersion != "v2" || p.Total != 25 {
		t.Fatalf("bad progress start: %+v", p)
	}
	p = waitFor(t, mg, "Widget")
	if p.Migrated+p.Skipped != 25 {
		t.Fatalf("not all objects processed: %+v", p)
	}
	if counts := st.CountByStorageVersion("Widget"); counts["v1"] != 0 || counts["v2"] != 25 {
		t.Fatalf("post-migration distribution wrong: %v", counts)
	}

	// every object must retain its v1-visible semantics when read as v1
	after, _ := st.List("Widget", "v1")
	if len(after) != 25 {
		t.Fatalf("object count changed: %d", len(after))
	}
	for _, o := range after {
		if o.Spec["name"] != before[o.Metadata.Name] {
			t.Fatalf("semantic drift on %s: %v != %v", o.Metadata.Name, o.Spec["name"], before[o.Metadata.Name])
		}
	}
	// and present proper v2 shapes when read as v2
	v2objs, _ := st.List("Widget", "v2")
	for _, o := range v2objs {
		if _, ok := o.Spec["display_name"]; !ok {
			t.Fatalf("v2 shape missing display_name: %#v", o.Spec)
		}
	}
	// removal gate: v1 is now drainable
	if err := mg.CanRemoveVersion("Widget", "v1"); err != nil {
		t.Fatalf("v1 should be removable after drain: %v", err)
	}
	// the registry's storage version has been switched
	k, _ := reg.Get("Widget")
	if k.StorageVersion != "v2" {
		t.Fatalf("registry storage version = %s, want v2", k.StorageVersion)
	}
}

func TestMigrationGateBlocksRemovalWhileObjectsRemain(t *testing.T) {
	mg, st, _, _ := buildStack(t)
	defer mg.Stop()
	seed(t, st, 5)
	if err := mg.CanRemoveVersion("Widget", "v1"); err == nil {
		t.Fatal("removal must be blocked while objects still use v1")
	}
}

func TestMigrationSurvivesConversionOutageAndWritesNoBadData(t *testing.T) {
	mg, st, svc, _ := buildStack(t)
	defer mg.Stop()
	seed(t, st, 10)

	p, err := mg.SwitchStorage("Widget", "v2")
	if err != nil {
		t.Fatal(err)
	}
	// let roughly half the objects migrate, then break the service
	time.Sleep(140 * time.Millisecond)
	svc.SetFault("Widget", convert.Fault{Down: true})
	time.Sleep(300 * time.Millisecond)

	// while down, traffic-facing writes fail and nothing is half-written
	obj := &types.Object{
		Group: "test.io", Kind: "Widget", Version: "v1",
		Metadata: types.ObjectMeta{Name: "new-during-outage", Namespace: "ns"},
		Spec:     map[string]any{"name": "x", "count": int64(1)},
	}
	if _, err := st.Create(obj); err == nil {
		t.Fatal("write during conversion outage must fail")
	}
	// all existing records must still decode cleanly in their stored shape
	for _, key := range st.KeysForVersion("Widget", "v1") {
		parts := strings.SplitN(key, "/", 2)
		sv, raw, ok := st.RawRecord("Widget", parts[0], parts[1])
		if !ok || raw == nil || sv == "" {
			t.Fatalf("corrupt record during outage: %s", key)
		}
	}

	// migration must not have completed while the service was down
	if prog := mg.Progress("Widget"); prog.State == StateCompleted {
		t.Fatal("migration completed despite conversion outage")
	}

	// recover: the backoff-retry loop drains the remaining objects
	svc.ClearFaults()
	p = waitFor(t, mg, "Widget")
	if p.FailedAttempts == 0 {
		t.Fatalf("expected failed attempts during outage, got %+v", p)
	}
	if st.CountByStorageVersion("Widget")["v1"] != 0 {
		t.Fatal("objects remain in v1 after recovery")
	}
}
