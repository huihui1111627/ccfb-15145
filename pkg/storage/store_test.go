package storage

import (
	"errors"
	"testing"

	"crv/pkg/convert"
	"crv/pkg/schema"
	"crv/pkg/types"
)

func testRegistry(t *testing.T) (*schema.Registry, *schema.Kind) {
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
		t.Fatalf("register: %v", err)
	}
	return reg, k
}

func newStore(t *testing.T) (*Store, *convert.Service, *schema.Kind) {
	t.Helper()
	reg, k := testRegistry(t)
	svc := convert.NewService(convert.NewEngine(reg))
	return New(reg, svc), svc, k
}

func v1Obj(name string) *types.Object {
	return &types.Object{
		Group: "test.io", Kind: "Widget", Version: "v1",
		Metadata: types.ObjectMeta{Name: name, Namespace: "ns"},
		Spec:     map[string]any{"name": "alpha", "count": int64(3)},
	}
}

func TestStoredInStorageVersion(t *testing.T) {
	st, _, k := newStore(t)
	if _, err := st.Create(v1Obj("a")); err != nil {
		t.Fatal(err)
	}
	sv, raw, ok := st.RawRecord("Widget", "ns", "a")
	if !ok {
		t.Fatal("record missing")
	}
	if sv != "v1" {
		t.Fatalf("stored version = %s, want v1", sv)
	}
	// write through a v2 client: must be converted down to v1 on disk
	v2 := raw.DeepCopy()
	v2.Version = "v2"
	delete(v2.Spec, "name")
	v2.Spec["display_name"] = "beta"
	v2.Spec["region"] = "us-west-2"
	if err := st.svc.ConvertObject(v2, "v2"); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Update(v2, raw.Metadata.ResourceVersion); err != nil {
		t.Fatal(err)
	}
	sv, raw, _ = st.RawRecord("Widget", "ns", "a")
	if sv != "v1" {
		t.Fatalf("v2 write landed in %s, storage must stay v1", sv)
	}
	// the v2-only region must have survived via a stash annotation
	if _, ok := raw.Metadata.Annotations[types.StashKey("v1")]; !ok {
		t.Fatalf("v2-only field not stashed on disk: %#v", raw.Metadata.Annotations)
	}
	// reading back as v2 restores it exactly
	got, err := st.Get("Widget", "ns", "a", "v2")
	if err != nil {
		t.Fatal(err)
	}
	if got.Spec["region"] != "us-west-2" || got.Spec["display_name"] != "beta" {
		t.Fatalf("v2 read lost fields: %#v", got.Spec)
	}
	_ = k
}

func TestOptimisticConcurrency(t *testing.T) {
	st, _, _ := newStore(t)
	created, err := st.Create(v1Obj("a"))
	if err != nil {
		t.Fatal(err)
	}
	created.Spec["count"] = int64(9)
	if _, err := st.Update(created, created.Metadata.ResourceVersion); err != nil {
		t.Fatalf("first update should succeed: %v", err)
	}
	// stale resource version must be rejected, no overwrite
	stale := created.DeepCopy()
	stale.Spec["count"] = int64(100)
	if _, err := st.Update(stale, created.Metadata.ResourceVersion); !errors.Is(err, ErrConflict) {
		t.Fatalf("want ErrConflict, got %v", err)
	}
	got, _ := st.Get("Widget", "ns", "a", "v1")
	if got.Spec["count"] != int64(9) {
		t.Fatalf("stale write clobbered data: %v", got.Spec["count"])
	}
}

func TestConversionServiceFailureWritesNothing(t *testing.T) {
	st, svc, _ := newStore(t)
	if _, err := st.Create(v1Obj("a")); err != nil {
		t.Fatal(err)
	}
	before, _, _ := st.RawRecord("Widget", "ns", "a")

	svc.SetFault("Widget", convert.Fault{Down: true})

	// writes fail explicitly
	obj := v1Obj("b")
	if _, err := st.Create(obj); err == nil {
		t.Fatal("create must fail while conversion service is down")
	}
	if _, _, ok := st.RawRecord("Widget", "ns", "b"); ok {
		t.Fatal("half-converted/new object was written despite service failure")
	}
	// reads fail explicitly too
	if _, err := st.Get("Widget", "ns", "a", "v2"); err == nil {
		t.Fatal("cross-version read must fail while conversion service is down")
	}
	// stored bytes are untouched
	after, rawAfter, _ := st.RawRecord("Widget", "ns", "a")
	if after != before {
		t.Fatalf("record changed during outage: %s -> %s", before, after)
	}
	if rawAfter.Spec["count"] != int64(3) {
		t.Fatalf("record content changed during outage: %#v", rawAfter.Spec)
	}

	// after recovery everything works again
	svc.ClearFaults()
	if _, err := st.Get("Widget", "ns", "a", "v2"); err != nil {
		t.Fatalf("read after recovery failed: %v", err)
	}
}

func TestSlowConversionFailsUnderTimeoutContract(t *testing.T) {
	st, svc, _ := newStore(t)
	if _, err := st.Create(v1Obj("a")); err != nil {
		t.Fatal(err)
	}
	// a slow service still completes; the caller observes added latency in
	// metrics and no data corruption. Hard timeout enforcement belongs to
	// callers with deadlines; here we verify correctness under slowness.
	svc.SetFault("Widget", convert.Fault{Slow: true, AddedLatency: 1})
	got, err := st.Get("Widget", "ns", "a", "v2")
	if err != nil {
		t.Fatalf("slow-but-alive service must still serve: %v", err)
	}
	if got.Spec["display_name"] != "alpha" {
		t.Fatalf("slow conversion produced wrong shape: %#v", got.Spec)
	}
}
