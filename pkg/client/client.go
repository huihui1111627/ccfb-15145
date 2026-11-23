// Package client implements versioned platform clients. Each client is
// pinned to one schema version and speaks only that version; the store
// converts on its behalf.
//
// Two client personalities model real-world behavior:
//
//   - PreservingClient (the normal K8s-style client): unknown annotations are
//     opaque metadata and are written back unchanged, so stashed new-version
//     fields survive an old client's read-modify-write.
//   - AggressiveClient: strips every annotation it does not itself recognize
//     on write. Even then, the storage layer re-applies the stash on the
//     downgrade path when the object is read again in a newer version only if
//     the annotations survived — so the acceptance suite uses this client to
//     document the exact boundary of the lossless guarantee.
package client

import (
	"crv/pkg/storage"
	"crv/pkg/types"
)

type Personality int

const (
	Preserving  Personality = iota // keeps unknown annotations
	Aggressive                     // drops annotations it didn't create
)

// VersionedClient pins all operations to one version of one kind.
type VersionedClient struct {
	store       *storage.Store
	kind        string
	version     string
	personality Personality
	knownAnno   map[string]bool // annotations the client itself set
}

func New(st *storage.Store, kind, version string, p Personality) *VersionedClient {
	return &VersionedClient{
		store: st, kind: kind, version: version,
		personality: p, knownAnno: map[string]bool{},
	}
}

func (c *VersionedClient) Version() string { return c.version }
func (c *VersionedClient) Kind() string    { return c.kind }

// Get reads the object in the client's version.
func (c *VersionedClient) Get(namespace, name string) (*types.Object, error) {
	return c.store.Get(c.kind, namespace, name, c.version)
}

// List reads all objects in the client's version.
func (c *VersionedClient) List() ([]*types.Object, error) {
	return c.store.List(c.kind, c.version)
}

// Create writes a new object in the client's version.
func (c *VersionedClient) Create(obj *types.Object) (*types.Object, error) {
	obj.Kind = c.kind
	obj.Version = c.version
	c.scrub(obj)
	return c.store.Create(obj)
}

// Update performs a read-modify-write with optimistic concurrency. The
// caller passes the object it got from Get (with its ResourceVersion) after
// mutating fields it understands.
func (c *VersionedClient) Update(obj *types.Object) (*types.Object, error) {
	obj.Kind = c.kind
	obj.Version = c.version
	expected := obj.Metadata.ResourceVersion
	c.scrub(obj)
	return c.store.Update(obj, expected)
}

// SetAnnotation records a client-owned annotation; Aggressive clients keep
// annotations they created and drop everything else.
func (c *VersionedClient) SetAnnotation(obj *types.Object, key, value string) {
	if obj.Metadata.Annotations == nil {
		obj.Metadata.Annotations = map[string]string{}
	}
	obj.Metadata.Annotations[key] = value
	c.knownAnno[key] = true
}

// scrub applies the client's personality to outgoing objects.
func (c *VersionedClient) scrub(obj *types.Object) {
	if c.personality != Aggressive || len(obj.Metadata.Annotations) == 0 {
		return
	}
	kept := map[string]string{}
	for k, v := range obj.Metadata.Annotations {
		if c.knownAnno[k] {
			kept[k] = v
		}
	}
	if len(kept) == 0 {
		obj.Metadata.Annotations = nil
	} else {
		obj.Metadata.Annotations = kept
	}
}
