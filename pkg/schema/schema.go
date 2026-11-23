// Package schema holds the simulated custom-resource definitions: each kind
// has several ordered versions, a set of field definitions per version, and
// declarative conversion rules between every pair of adjacent versions.
package schema

import (
	"fmt"
	"sort"
)

// FieldType enumerates the leaf types the simulated resources use.
type FieldType string

const (
	TypeString FieldType = "string"
	TypeInt    FieldType = "int"
	TypeBool   FieldType = "bool"
	TypeFloat  FieldType = "float"
	TypeList   FieldType = "list"  // list of strings
	TypeObject FieldType = "object" // nested object
)

// FieldDef describes one field path of one version, e.g.
// {Path: "spec.replica", Type: TypeInt}.
type FieldDef struct {
	Path string    `json:"path"`
	Type FieldType `json:"type"`
	// Optional marks fields a converter is allowed to synthesize with a
	// zero value when upgrading from a version that lacks them.
	Optional bool `json:"optional,omitempty"`
}

// VersionSchema is the complete field set a client speaking this version
// understands.
type VersionSchema struct {
	Name   string     `json:"name"`
	Fields []FieldDef `json:"fields"`
	// Deprecated versions still serve traffic but may be removed once no
	// stored object remains in them.
	Deprecated bool `json:"deprecated,omitempty"`
}

// RuleKind enumerates the supported adjacent-version evolution operations.
// All rules are declared on the older→newer edge; the engine derives their
// inverse automatically for the newer→older direction.
type RuleKind string

const (
	// RuleRename: a field was renamed. Source/sink are leaf paths.
	RuleRename RuleKind = "rename"
	// RuleAdd: a field introduced in the newer version, with a default
	// synthesized on upgrade. On downgrade it is stashed.
	RuleAdd RuleKind = "add"
	// RuleSplit: one field in the old version becomes several fields in
	// the new version (e.g. host:port -> host + port).
	RuleSplit RuleKind = "split"
	// RuleMerge: inverse of split; declared only when split needs custom
	// join formatting (handled as split with MergeSep).
	// RuleWrapList: a scalar field became a list (single value -> [v]).
	RuleWrapList RuleKind = "wrap_list"
)

// ConversionRule is one evolution step on an adjacent version edge.
type ConversionRule struct {
	Kind RuleKind `json:"kind"`
	// Source is the old-version path; Sink the new-version path. For
	// split, Sources carries the expanded old→new paths instead.
	Source string   `json:"source,omitempty"`
	Sink   string   `json:"sink,omitempty"`
	// Add rules: default value to synthesize when upgrading.
	Default any      `json:"default,omitempty"`
	// Split rules: parallel path pairs, e.g. spec.endpoint -> spec.host, spec.port.
	Sources [][2]string `json:"sources,omitempty"`
	// MergeSep joins split strings back on downgrade ("host:port" => ":").
	MergeSep string `json:"merge_sep,omitempty"`
}

// Kind is one simulated resource type: ordered versions + edge rules.
type Kind struct {
	Group          string             `json:"group"`
	Name           string             `json:"name"`
	Versions       []*VersionSchema   `json:"versions"`
	StorageVersion string             `json:"storage_version"`
	// Rules[fromVersion] = rules applied when converting fromVersion -> next.
	Rules map[string][]*ConversionRule `json:"rules"`
}

// VersionIndex returns the ordinal position of a version (older < newer).
func (k *Kind) VersionIndex(v string) (int, error) {
	for i, vs := range k.Versions {
		if vs.Name == v {
			return i, nil
		}
	}
	return -1, fmt.Errorf("kind %s has no version %q", k.Name, v)
}

// Version returns the schema for one version.
func (k *Kind) Version(v string) (*VersionSchema, error) {
	for _, vs := range k.Versions {
		if vs.Name == v {
			return vs, nil
		}
	}
	return nil, fmt.Errorf("kind %s has no version %q", k.Name, v)
}

// FieldSet returns the known field paths of a version, including the
// intermediate container paths (e.g. "spec" itself).
func (k *Kind) FieldSet(v string) (map[string]bool, error) {
	vs, err := k.Version(v)
	if err != nil {
		return nil, err
	}
	set := map[string]bool{
		"spec":   true,
		"status": true,
	}
	for _, f := range vs.Fields {
		set[f.Path] = true
		// every prefix of a nested path is a known container
		p := ""
		for _, seg := range splitPath(f.Path) {
			if p == "" {
				p = seg
			} else {
				p += "." + seg
			}
			set[p] = true
		}
	}
	return set, nil
}

// Registry is the in-memory catalogue of all resource definitions.
type Registry struct {
	kinds map[string]*Kind
}

func NewRegistry() *Registry {
	return &Registry{kinds: map[string]*Kind{}}
}

func (r *Registry) Register(k *Kind) error {
	if _, ok := r.kinds[k.Name]; ok {
		return fmt.Errorf("kind %s already registered", k.Name)
	}
	if _, err := k.VersionIndex(k.StorageVersion); err != nil {
		return fmt.Errorf("kind %s storage version invalid: %w", k.Name, err)
	}
	// every rules edge must reference adjacent known versions
	for from := range k.Rules {
		fi, err := k.VersionIndex(from)
		if err != nil {
			return err
		}
		if fi >= len(k.Versions)-1 {
			return fmt.Errorf("kind %s rules on non-adjacent edge %s", k.Name, from)
		}
		if k.Versions[fi+1].Name != nextVersion(k, from) {
			return fmt.Errorf("kind %s rules edge %q not adjacent", k.Name, from)
		}
	}
	r.kinds[k.Name] = k
	return nil
}

func nextVersion(k *Kind, from string) string {
	fi, _ := k.VersionIndex(from)
	return k.Versions[fi+1].Name
}

// Get returns a kind definition.
func (r *Registry) Get(name string) (*Kind, error) {
	k, ok := r.kinds[name]
	if !ok {
		return nil, fmt.Errorf("unknown kind %q", name)
	}
	return k, nil
}

// All returns kinds sorted by name for stable UI/report ordering.
func (r *Registry) All() []*Kind {
	out := make([]*Kind, 0, len(r.kinds))
	for _, k := range r.kinds {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// SetStorageVersion rewrites the configured storage version. The migrator
// is responsible for rewriting existing objects; new writes immediately
// land in the new version.
func (r *Registry) SetStorageVersion(kind, version string) error {
	k, err := r.Get(kind)
	if err != nil {
		return err
	}
	if _, err := k.VersionIndex(version); err != nil {
		return err
	}
	k.StorageVersion = version
	return nil
}

// RemoveVersion drops a version from a kind's definition. It is only legal
// after storage migration has fully drained that version.
func (r *Registry) RemoveVersion(kind, version string) error {
	k, err := r.Get(kind)
	if err != nil {
		return err
	}
	idx, err := k.VersionIndex(version)
	if err != nil {
		return err
	}
	if k.StorageVersion == version {
		return fmt.Errorf("cannot remove %s: it is the storage version of %s", version, kind)
	}
	if version == k.Versions[0].Name || version == k.Versions[len(k.Versions)-1].Name {
		return fmt.Errorf("cannot remove endpoint version %s of kind %s", version, kind)
	}
	k.Versions = append(k.Versions[:idx], k.Versions[idx+1:]...)
	delete(k.Rules, version)
	// Re-anchor rule edges: rules were keyed by the older version name.
	// Removing an interior version merges its two edges; the caller is
	// expected to only remove the oldest served version in practice.
	return nil
}

func splitPath(p string) []string {
	var segs []string
	start := 0
	for i := 0; i < len(p); i++ {
		if p[i] == '.' {
			segs = append(segs, p[start:i])
			start = i + 1
		}
	}
	return append(segs, p[start:])
}
