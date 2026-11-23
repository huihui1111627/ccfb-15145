// Package types defines the generic custom-resource object model used across
// the platform. An object is versioned: every client request carries the
// version of the schema it speaks, while the storage layer persists objects
// in the kind's configured storage version.
package types

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// StashAnnotationPrefix is the annotation key prefix under which fields that
// cannot be represented in the target version are hidden. The full key is
// "crv.io/stash.<version>", e.g. crv.io/stash.v1.
const StashAnnotationPrefix = "crv.io/stash."

// Object is a versioned custom resource. Spec and Status are free-form maps
// described by the schema registry; the conversion engine treats every leaf
// path (e.g. "spec.replica") as a field.
type Object struct {
	Group     string         `json:"group"`
	Kind      string         `json:"kind"`
	Version   string         `json:"version"`
	Metadata  ObjectMeta     `json:"metadata"`
	Spec      map[string]any `json:"spec"`
	Status    map[string]any `json:"status"`
}

// ObjectMeta carries identity, concurrency control and annotations.
type ObjectMeta struct {
	Name            string            `json:"name"`
	Namespace       string            `json:"namespace"`
	ResourceVersion int64             `json:"resourceVersion"`
	Labels          map[string]string `json:"labels,omitempty"`
	Annotations     map[string]string `json:"annotations,omitempty"`
}

// Key returns the storage identity of an object (kind is namespaced per
// group in the registry, so namespace/name is enough inside one kind).
func (o *Object) Key() string {
	return o.Metadata.Namespace + "/" + o.Metadata.Name
}

// DeepCopy returns an independent copy. All persisted and returned objects
// are deep copies so callers can never mutate storage state directly.
func (o *Object) DeepCopy() *Object {
	cp := *o
	cp.Metadata = deepCopyMeta(o.Metadata)
	cp.Spec = deepCopyMap(o.Spec)
	cp.Status = deepCopyMap(o.Status)
	return &cp
}

func deepCopyMeta(m ObjectMeta) ObjectMeta {
	cp := m
	if m.Labels != nil {
		cp.Labels = make(map[string]string, len(m.Labels))
		for k, v := range m.Labels {
			cp.Labels[k] = v
		}
	}
	if m.Annotations != nil {
		cp.Annotations = make(map[string]string, len(m.Annotations))
		for k, v := range m.Annotations {
			cp.Annotations[k] = v
		}
	}
	return cp
}

func deepCopyMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = deepCopyValue(v)
	}
	return out
}

func deepCopyValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		return deepCopyMap(t)
	case []any:
		out := make([]any, len(t))
		for i := range t {
			out[i] = deepCopyValue(t[i])
		}
		return out
	default:
		return v
	}
}

// Encode serializes an object. json.Number is used on decode so integer
// fields survive an annotation round-trip without turning into float64.
func Encode(o *Object) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(o); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Decode parses an object, keeping all numbers as json.Number.
func Decode(data []byte) (*Object, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var o Object
	if err := dec.Decode(&o); err != nil {
		return nil, err
	}
	normalizeNumbers(o.Spec)
	normalizeNumbers(o.Status)
	return &o, nil
}

// normalizeNumbers turns json.Number leaves into int64 when integral so
// deep-equality comparisons see stable types across stash round-trips.
func normalizeNumbers(m map[string]any) {
	for k, v := range m {
		switch t := v.(type) {
		case json.Number:
			m[k] = normalizeNumber(t)
		case map[string]any:
			normalizeNumbers(t)
		case []any:
			for i, e := range t {
				switch et := e.(type) {
				case json.Number:
					t[i] = normalizeNumber(et)
				case map[string]any:
					normalizeNumbers(et)
				}
			}
		}
	}
}

func normalizeNumber(n json.Number) any {
	if i, err := n.Int64(); err == nil {
		return i
	}
	f, err := n.Float64()
	if err != nil {
		return n.String()
	}
	return f
}

// StashKey returns the annotation key used for fields hidden when leaving
// the given version (and restored when returning to it).
func StashKey(version string) string {
	return StashAnnotationPrefix + version
}

// String gives a stable human identity for logs and errors.
func (o *Object) String() string {
	return fmt.Sprintf("%s/%s %s/%s rv=%d", o.Group, o.Kind, o.Metadata.Namespace, o.Metadata.Name, o.Metadata.ResourceVersion)
}
