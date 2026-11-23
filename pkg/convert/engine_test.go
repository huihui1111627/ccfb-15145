package convert

import (
	"reflect"
	"testing"

	"crv/pkg/schema"
	"crv/pkg/types"
)

// testKind has 3 versions covering every evolution rule type:
//
//	v1 -> v2: rename (spec.name -> spec.display_name),
//	          add    (spec.priority, default 5),
//	          wrap   (spec.tag scalar -> spec.tag_list)
//	v2 -> v3: split  (spec.endpoint -> spec.endpoint_host + spec.endpoint_port),
//	          add    (spec.encrypted, default false)
func testKind() *schema.Kind {
	return &schema.Kind{
		Group:          "test.io",
		Name:           "Widget",
		StorageVersion: "v1",
		Versions: []*schema.VersionSchema{
			{Name: "v1", Fields: []schema.FieldDef{
				{Path: "spec.name", Type: schema.TypeString},
				{Path: "spec.tag", Type: schema.TypeString},
				{Path: "spec.endpoint", Type: schema.TypeString},
				{Path: "spec.shared", Type: schema.TypeInt},
				{Path: "status.phase", Type: schema.TypeString},
			}},
			{Name: "v2", Fields: []schema.FieldDef{
				{Path: "spec.display_name", Type: schema.TypeString},
				{Path: "spec.priority", Type: schema.TypeInt, Optional: true},
				{Path: "spec.tag_list", Type: schema.TypeList},
				{Path: "spec.endpoint", Type: schema.TypeString},
				{Path: "spec.shared", Type: schema.TypeInt},
				{Path: "status.phase", Type: schema.TypeString},
			}},
			{Name: "v3", Fields: []schema.FieldDef{
				{Path: "spec.display_name", Type: schema.TypeString},
				{Path: "spec.priority", Type: schema.TypeInt, Optional: true},
				{Path: "spec.tag_list", Type: schema.TypeList},
				{Path: "spec.endpoint_host", Type: schema.TypeString},
				{Path: "spec.endpoint_port", Type: schema.TypeString},
				{Path: "spec.encrypted", Type: schema.TypeBool, Optional: true},
				{Path: "spec.shared", Type: schema.TypeInt},
				{Path: "status.phase", Type: schema.TypeString},
			}},
		},
		Rules: map[string][]*schema.ConversionRule{
			"v1": {
				{Kind: schema.RuleRename, Source: "spec.name", Sink: "spec.display_name"},
				{Kind: schema.RuleAdd, Sink: "spec.priority", Default: int64(5)},
				{Kind: schema.RuleWrapList, Source: "spec.tag", Sink: "spec.tag_list"},
			},
			"v2": {
				{Kind: schema.RuleSplit, Sources: [][2]string{
					{"spec.endpoint", "spec.endpoint_host"},
					{"spec.endpoint", "spec.endpoint_port"},
				}, MergeSep: ":"},
				{Kind: schema.RuleAdd, Sink: "spec.encrypted", Default: false},
			},
		},
	}
}

func newEngine(t *testing.T) (*Engine, *schema.Kind) {
	t.Helper()
	reg := schema.NewRegistry()
	k := testKind()
	if err := reg.Register(k); err != nil {
		t.Fatalf("register: %v", err)
	}
	return NewEngine(reg), k
}

func v1Object() *types.Object {
	return &types.Object{
		Group: "test.io", Kind: "Widget", Version: "v1",
		Metadata: types.ObjectMeta{Name: "w1", Namespace: "ns"},
		Spec: map[string]any{
			"name":     "widget-a",
			"tag":      "blue",
			"endpoint": "db.example.com:5432",
			"shared":   int64(42),
		},
		Status: map[string]any{"phase": "Running"},
	}
}

func TestUpgradeDowngradeShape(t *testing.T) {
	e, _ := newEngine(t)

	up := v1Object()
	if err := e.Convert(up, "v3"); err != nil {
		t.Fatalf("convert v1->v3: %v", err)
	}
	if up.Version != "v3" {
		t.Fatalf("version = %s, want v3", up.Version)
	}
	wantSpec := map[string]any{
		"display_name":  "widget-a",
		"priority":      int64(5),
		"tag_list":      []any{"blue"},
		"endpoint_host": "db.example.com",
		"endpoint_port": "5432",
		"encrypted":     false,
		"shared":        int64(42),
	}
	if !reflect.DeepEqual(up.Spec, wantSpec) {
		t.Fatalf("v3 spec mismatch:\n got %#v\nwant %#v", up.Spec, wantSpec)
	}
	if up.Status["phase"] != "Running" {
		t.Fatalf("status lost: %#v", up.Status)
	}

	back := up.DeepCopy()
	if err := e.Convert(back, "v1"); err != nil {
		t.Fatalf("convert v3->v1: %v", err)
	}
	// fields v1 knows come back as v1 shapes
	if back.Spec["name"] != "widget-a" {
		t.Fatalf("rename not reversed: %#v", back.Spec)
	}
	if back.Spec["tag"] != "blue" {
		t.Fatalf("wrap not reversed: %#v", back.Spec)
	}
	if back.Spec["endpoint"] != "db.example.com:5432" {
		t.Fatalf("split merge = %v", back.Spec["endpoint"])
	}
	// Fields v1 cannot know are either stashed or deterministically
	// rebuildable (add-rule defaults, single-element wrapped list). In
	// either case converting back to v3 must restore the exact v3 shape.
	reUp := back.DeepCopy()
	if err := e.Convert(reUp, "v3"); err != nil {
		t.Fatalf("re-upgrade v1->v3: %v", err)
	}
	if !reflect.DeepEqual(reUp.Spec, wantSpec) {
		t.Fatalf("v3 shape not restored after downgrade:\n got %#v\nwant %#v", reUp.Spec, wantSpec)
	}
}

func TestRoundTripAllPairs(t *testing.T) {
	e, k := newEngine(t)
	versions := []string{"v1", "v2", "v3"}

	for _, start := range versions {
		for _, middle := range versions {
			t.Run(start+"->"+middle+"->"+start, func(t *testing.T) {
				// build a fully populated object of the start version
				obj, err := fullObject(k, start, 1)
				if err != nil {
					t.Fatal(err)
				}
				original := obj.DeepCopy()

				if err := e.Convert(obj, middle); err != nil {
					t.Fatalf("forward convert: %v", err)
				}
				if err := e.Convert(obj, start); err != nil {
					t.Fatalf("back convert: %v", err)
				}
				if !reflect.DeepEqual(obj.Spec, original.Spec) {
					t.Fatalf("spec not lossless\n got %#v\nwant %#v", obj.Spec, original.Spec)
				}
				if !reflect.DeepEqual(obj.Status, original.Status) {
					t.Fatalf("status not lossless\n got %#v\nwant %#v", obj.Status, original.Status)
				}
				if !reflect.DeepEqual(obj.Metadata.Annotations, original.Metadata.Annotations) {
					t.Fatalf("annotations not restored\n got %#v\nwant %#v",
						obj.Metadata.Annotations, original.Metadata.Annotations)
				}
			})
		}
	}
}

func TestMultiValueListSurvivesDowngrade(t *testing.T) {
	e, _ := newEngine(t)
	// v3 object whose tag_list has several elements — v1 only has scalar tag
	obj := v1Object()
	if err := e.Convert(obj, "v3"); err != nil {
		t.Fatal(err)
	}
	obj.Spec["tag_list"] = []any{"blue", "green", "red"}

	back := obj.DeepCopy()
	if err := e.Convert(back, "v1"); err != nil {
		t.Fatal(err)
	}
	if back.Spec["tag"] != "blue" {
		t.Fatalf("scalar view = %v, want blue", back.Spec["tag"])
	}
	upAgain := back.DeepCopy()
	if err := e.Convert(upAgain, "v3"); err != nil {
		t.Fatal(err)
	}
	got, ok := upAgain.Spec["tag_list"].([]any)
	if !ok || !reflect.DeepEqual(got, []any{"blue", "green", "red"}) {
		t.Fatalf("multi-value list lost in round-trip: %#v", upAgain.Spec["tag_list"])
	}
}

func TestClientAnnotationsArePreserved(t *testing.T) {
	e, _ := newEngine(t)
	obj := v1Object()
	obj.Metadata.Annotations = map[string]string{"ops.io/owner": "team-a"}
	if err := e.Convert(obj, "v3"); err != nil {
		t.Fatal(err)
	}
	if obj.Metadata.Annotations["ops.io/owner"] != "team-a" {
		t.Fatalf("client annotation clobbered: %#v", obj.Metadata.Annotations)
	}
	back := obj.DeepCopy()
	if err := e.Convert(back, "v1"); err != nil {
		t.Fatal(err)
	}
	if back.Metadata.Annotations["ops.io/owner"] != "team-a" {
		t.Fatalf("client annotation lost on downgrade: %#v", back.Metadata.Annotations)
	}
}

func TestUnknownVersionFailsCleanly(t *testing.T) {
	e, _ := newEngine(t)
	obj := v1Object()
	if err := e.Convert(obj, "v9"); err == nil {
		t.Fatal("expected error for unknown version")
	}
	if obj.Version != "v1" {
		t.Fatalf("failed conversion mutated object version: %s", obj.Version)
	}
}
