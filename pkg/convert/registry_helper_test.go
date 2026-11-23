package convert

import (
	"testing"

	"crv/pkg/schema"
)

func newTestRegistry(t *testing.T) *schema.Registry {
	t.Helper()
	reg := schema.NewRegistry()
	if err := reg.Register(testKind()); err != nil {
		t.Fatalf("register: %v", err)
	}
	return reg
}

func regMustGet(t *testing.T, reg *schema.Registry, name string) *schema.Kind {
	t.Helper()
	k, err := reg.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	return k
}
