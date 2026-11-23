package convert

import "testing"

// TestSplitPreservesExtraSeparators covers a value that contains more
// separators than the split has target fields: cutting must be capped so the
// final piece keeps its embedded separators and the merge restores the
// exact original string through the whole version chain.
func TestSplitPreservesExtraSeparators(t *testing.T) {
	reg := newTestRegistry(t)
	e := NewEngine(reg)
	k := regMustGet(t, reg, "Widget")

	obj := v1Object()
	obj.Spec["endpoint"] = "helix-374:alpha-86:falcon-481"

	up := obj.DeepCopy()
	if err := e.Convert(up, "v3"); err != nil {
		t.Fatal(err)
	}
	if up.Spec["endpoint_host"] != "helix-374" {
		t.Fatalf("host = %v, want helix-374", up.Spec["endpoint_host"])
	}
	if up.Spec["endpoint_port"] != "alpha-86:falcon-481" {
		t.Fatalf("port must retain embedded separator, got %v", up.Spec["endpoint_port"])
	}

	back := up.DeepCopy()
	if err := e.Convert(back, "v1"); err != nil {
		t.Fatal(err)
	}
	if back.Spec["endpoint"] != "helix-374:alpha-86:falcon-481" {
		t.Fatalf("endpoint round-trip = %v", back.Spec["endpoint"])
	}

	// and the full chain back up must still be exact
	again := back.DeepCopy()
	if err := e.Convert(again, "v3"); err != nil {
		t.Fatal(err)
	}
	if again.Spec["endpoint_port"] != "alpha-86:falcon-481" {
		t.Fatalf("second-upgrade port = %v", again.Spec["endpoint_port"])
	}
	_ = k
}
