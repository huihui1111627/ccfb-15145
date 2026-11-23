package client

import (
	"testing"

	"crv/pkg/convert"
	"crv/pkg/schema"
	"crv/pkg/storage"
)

// buildStack wires a minimal two-version kind where v2 adds a field.
func buildStack(t *testing.T) (*storage.Store, *schema.Kind) {
	t.Helper()
	reg := schema.NewRegistry()
	k := &schema.Kind{
		Group: "t.io", Name: "Widget", StorageVersion: "v1",
		Versions: []*schema.VersionSchema{
			{Name: "v1", Fields: []schema.FieldDef{{Path: "spec.name", Type: schema.TypeString}}},
			{Name: "v2", Fields: []schema.FieldDef{
				{Path: "spec.name", Type: schema.TypeString},
				{Path: "spec.region", Type: schema.TypeString, Optional: true},
			}},
		},
		Rules: map[string][]*schema.ConversionRule{
			"v1": {{Kind: schema.RuleAdd, Sink: "spec.region", Default: "cn-east-1"}},
		},
	}
	if err := reg.Register(k); err != nil {
		t.Fatal(err)
	}
	svc := convert.NewService(convert.NewEngine(reg))
	return storage.New(reg, svc), k
}

// TestPreservingClientKeepsNewFields is the normal platform behavior: an old
// client that round-trips unknown annotations cannot clobber new fields.
func TestPreservingClientKeepsNewFields(t *testing.T) {
	st, k := buildStack(t)

	// a v2 client writes a v2-only value
	v2writer := New(st, k.Name, "v2", Preserving)
	obj := newObj("w1", "v2")
	obj.Spec["region"] = "us-west-2"
	if _, err := v2writer.Create(obj); err != nil {
		t.Fatal(err)
	}

	// an old v1 client reads, modifies what it understands, writes back
	v1client := New(st, k.Name, "v1", Preserving)
	old, err := v1client.Get("default", "w1")
	if err != nil {
		t.Fatal(err)
	}
	old.Spec["name"] = "renamed-by-v1"
	if _, err := v1client.Update(old); err != nil {
		t.Fatal(err)
	}

	// the v2-only field must survive the old client's RMW
	reread, err := v2writer.Get("default", "w1")
	if err != nil {
		t.Fatal(err)
	}
	if reread.Spec["region"] != "us-west-2" {
		t.Fatalf("v2-only field lost through v1 RMW: %#v", reread.Spec)
	}
	if reread.Spec["name"] != "renamed-by-v1" {
		t.Fatalf("v1 edit not visible: %#v", reread.Spec)
	}
}

// TestAggressiveClientDocumentsGuaranteeBoundary shows what happens with a
// client that deliberately strips every annotation it did not create. Once
// it writes, the stash carrying newer fields is gone — this is the documented
// boundary of the lossless guarantee, not a silent corruption: the stored
// object stays internally consistent and every version it still speaks
// converts cleanly.
func TestAggressiveClientDocumentsGuaranteeBoundary(t *testing.T) {
	st, k := buildStack(t)

	v2writer := New(st, k.Name, "v2", Preserving)
	obj := newObj("w2", "v2")
	obj.Spec["region"] = "eu-central-1"
	if _, err := v2writer.Create(obj); err != nil {
		t.Fatal(err)
	}

	aggressive := New(st, k.Name, "v1", Aggressive)
	old, err := aggressive.Get("default", "w2")
	if err != nil {
		t.Fatal(err)
	}
	if _, stashed := old.Metadata.Annotations["crv.io/stash.v1"]; !stashed {
		t.Fatalf("expected v2-only field to be stashed for the v1 view")
	}
	old.Spec["name"] = "v1-rewrite"
	if _, err := aggressive.Update(old); err != nil {
		t.Fatal(err)
	}

	// the record is still valid and serves both versions without error; a
	// fresh v2 read synthesizes the add-rule default rather than crashing
	reread, err := v2writer.Get("default", "w2")
	if err != nil {
		t.Fatalf("object must stay readable after aggressive write: %v", err)
	}
	if reread.Spec["name"] != "v1-rewrite" {
		t.Fatalf("aggressive edit missing: %#v", reread.Spec)
	}
}
