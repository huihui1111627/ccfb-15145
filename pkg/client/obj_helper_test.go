package client

import "crv/pkg/types"

func newObj(name, version string) *types.Object {
	return &types.Object{
		Group: "t.io", Kind: "Widget", Version: version,
		Metadata: types.ObjectMeta{Name: name, Namespace: "default"},
		Spec:     map[string]any{"name": "initial"},
	}
}
