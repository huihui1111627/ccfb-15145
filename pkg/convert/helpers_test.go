package convert

import (
	"fmt"

	"crv/pkg/schema"
	"crv/pkg/types"
)

// fullObject populates every field of one version with deterministic values.
func fullObject(k *schema.Kind, version string, seed int) (*types.Object, error) {
	vs, err := k.Version(version)
	if err != nil {
		return nil, err
	}
	obj := &types.Object{
		Group: k.Group, Kind: k.Name, Version: version,
		Metadata: types.ObjectMeta{
			Name: "full", Namespace: "ns",
			Annotations: map[string]string{"ops.io/owner": "team-a"},
		},
		Spec:   map[string]any{},
		Status: map[string]any{},
	}
	for i, f := range vs.Fields {
		put(obj, f.Path, sampleValue(f.Type, seed+i))
	}
	return obj, nil
}

func sampleValue(t schema.FieldType, n int) any {
	switch t {
	case schema.TypeString:
		return fmt.Sprintf("str-%d", n)
	case schema.TypeInt:
		return int64(100 + n)
	case schema.TypeBool:
		return n%2 == 0
	case schema.TypeFloat:
		return float64(n) * 1.5
	case schema.TypeList:
		return []any{fmt.Sprintf("a-%d", n), fmt.Sprintf("b-%d", n)}
	case schema.TypeObject:
		return map[string]any{"inner": fmt.Sprintf("o-%d", n)}
	default:
		return nil
	}
}
func put(obj *types.Object, path string, val any) {
	segs := splitDotted(path)
	node := sectionMap(obj, segs[0])
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
