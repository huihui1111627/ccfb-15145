// Package convert implements version conversion for generic custom
// resources.
//
// Conversion rules are declared only between adjacent versions; converting
// between arbitrary versions walks the version chain one edge at a time.
// Every field the target version cannot represent is serialized into an
// annotation ("crv.io/stash.<version>") instead of being dropped, which
// makes every conversion round-trip lossless: a read-modify-write performed
// by an old client cannot clobber fields introduced by newer versions.
package convert

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"crv/pkg/schema"
	"crv/pkg/types"
)

// Engine converts objects between any two versions of a registered kind.
type Engine struct {
	reg *schema.Registry
}

func NewEngine(reg *schema.Registry) *Engine {
	return &Engine{reg: reg}
}

// Convert rewrites obj in place so it is shaped like targetVersion. The
// object's Version field is updated on success. A failure leaves the object
// untouched: callers must never persist a partially converted object.
func (e *Engine) Convert(obj *types.Object, targetVersion string) error {
	if obj == nil {
		return fmt.Errorf("cannot convert nil object")
	}
	k, err := e.reg.Get(obj.Kind)
	if err != nil {
		return err
	}
	from, err := k.VersionIndex(obj.Version)
	if err != nil {
		return err
	}
	to, err := k.VersionIndex(targetVersion)
	if err != nil {
		return err
	}
	if from == to {
		return nil
	}
	// Work on a scratch copy so an error mid-chain never leaves the
	// caller's object half converted.
	work := obj.DeepCopy()
	if from < to {
		for i := from; i < to; i++ {
			if err := e.stepForward(k, k.Versions[i].Name, k.Versions[i+1].Name, work); err != nil {
				return fmt.Errorf("conversion %s %s->%s failed: %w", k.Name, k.Versions[i].Name, k.Versions[i+1].Name, err)
			}
		}
	} else {
		for i := from; i > to; i-- {
			if err := e.stepBackward(k, k.Versions[i].Name, k.Versions[i-1].Name, work); err != nil {
				return fmt.Errorf("conversion %s %s->%s failed: %w", k.Name, k.Versions[i].Name, k.Versions[i-1].Name, err)
			}
		}
	}
	work.Version = targetVersion
	*obj = *work
	return nil
}

// stepForward converts along one edge from older -> newer.
func (e *Engine) stepForward(k *schema.Kind, from, to string, obj *types.Object) error {
	// 1. Bring back data hidden the last time an object of this shape was
	//    downgraded away from this version. Types are restored from the
	//    target schema because JSON stash storage flattens ints to strings.
	restoreStash(k, from, obj)
	// 2. Apply the declared evolution rules.
	for _, r := range k.Rules[from] {
		if err := applyForwardRule(r, obj); err != nil {
			return err
		}
	}
	// 3. Anything still unknown to the target version is hidden, not lost.
	//    Add-rule defaults synthesized on this edge are dropped instead.
	if err := sweepIntoStash(k, from, to, obj); err != nil {
		return err
	}
	pruneAbsorbedStash(k, to, obj)
	return nil
}

// stepBackward converts along one edge from newer -> older.
func (e *Engine) stepBackward(k *schema.Kind, from, to string, obj *types.Object) error {
	restoreStash(k, from, obj)
	for _, r := range k.Rules[to] { // rules are keyed on the older edge version
		applyBackwardRule(r, obj)
	}
	// Fields added on edge to->from are unknown to the old version and are
	// hidden wholesale — including non-default user values. No defaults are
	// discarded here: unlike an upgrade sweep, a re-upgrade cannot assume a
	// hidden value still equals the rule default.
	if err := sweepIntoStash(k, "", to, obj); err != nil {
		return err
	}
	pruneAbsorbedStash(k, to, obj)
	return nil
}

// ---------------------------------------------------------------- rules --

func applyForwardRule(r *schema.ConversionRule, obj *types.Object) error {
	switch r.Kind {
	case schema.RuleRename:
		if v, ok := getPath(obj, r.Source); ok {
			if _, conflict := getPath(obj, r.Sink); !conflict {
				setPath(obj, r.Sink, v)
			}
			deletePath(obj, r.Source)
		}
	case schema.RuleAdd:
		if _, ok := getPath(obj, r.Sink); !ok {
			setPath(obj, r.Sink, r.Default)
		}
	case schema.RuleSplit:
		// A split field holds one string; on upgrade it is cut on the merge
		// separator into one piece per target path. SplitN caps the cut at
		// the target count, so separators inside what becomes the final
		// piece are preserved verbatim — splitting "a:b:c" into two fields
		// yields "a" and "b:c", and merging restores "a:b:c" exactly.
		raw, ok := getPath(obj, r.Sources[0][0])
		if !ok {
			return nil
		}
		joined, _ := raw.(string)
		var pieces []string
		if r.MergeSep != "" && strings.Contains(joined, r.MergeSep) {
			pieces = strings.SplitN(joined, r.MergeSep, len(r.Sources))
		} else {
			// No separator: the value predates split-aware writers. The
			// whole value goes to the first target and remaining targets
			// get empty strings; mergeSplitValues treats an empty tail as
			// "no second part", so downgrading restores the exact original.
			pieces = make([]string, len(r.Sources))
			pieces[0] = joined
		}
		for i, pair := range r.Sources {
			if i < len(pieces) {
				if _, conflict := getPath(obj, pair[1]); !conflict {
					setPath(obj, pair[1], pieces[i])
				}
			}
		}
		deletePath(obj, r.Sources[0][0])
	case schema.RuleWrapList:
		if v, ok := getPath(obj, r.Source); ok {
			if _, conflict := getPath(obj, r.Sink); !conflict {
				setPath(obj, r.Sink, []any{v})
			}
			deletePath(obj, r.Source)
		}
	}
	return nil
}

func applyBackwardRule(r *schema.ConversionRule, obj *types.Object) {
	// Values are copied back into the old-version paths; source paths that
	// are unknown to the old version are then removed here only when the
	// inverse was fully representable. Paths the old version cannot name
	// are left in place for the stash sweep to preserve losslessly.
	switch r.Kind {
	case schema.RuleRename:
		if v, ok := getPath(obj, r.Sink); ok {
			if _, conflict := getPath(obj, r.Source); !conflict {
				setPath(obj, r.Source, v)
				deletePath(obj, r.Sink) // both paths exist in both adjacent schemas
			}
		}
	case schema.RuleAdd:
		// inverse is entirely stash-driven: the added field is unknown to
		// the older version and gets hidden by the sweep.
	case schema.RuleSplit:
		values := make([]any, 0, len(r.Sources))
		allPresent := true
		for _, pair := range r.Sources {
			v, ok := getPath(obj, pair[1])
			if !ok {
				allPresent = false
				break
			}
			values = append(values, v)
		}
		if allPresent {
			// Recreate the old joined field unconditionally: this is the
			// inverse edge rebuilding a shape, not a client write, so an
			// existing value (e.g. one restored from a stash) is the same
			// data and must be replaced by the canonical merged form.
			setPath(obj, r.Sources[0][0], mergeSplitValues(values, r.MergeSep))
			for _, pair := range r.Sources {
				deletePath(obj, pair[1])
			}
		}
	case schema.RuleWrapList:
		if v, ok := getPath(obj, r.Sink); ok {
			if list, isList := v.([]any); isList && len(list) > 0 {
				if _, conflict := getPath(obj, r.Source); !conflict {
					setPath(obj, r.Source, list[0])
				}
			}
			// The list path is unknown to the old version, so the sweep
			// hides the whole list (extra elements included); on the next
			// upgrade it is restored and the multi-value list is intact.
		}
	}
}

func mergeSplitValues(values []any, sep string) any {
	if sep == "" {
		return values[0]
	}
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmt.Sprintf("%v", v)
	}
	// Empty trailing pieces mean the original value carried no separator;
	// dropping them (and the separator) restores the exact original string.
	for len(parts) > 1 && parts[len(parts)-1] == "" {
		parts = parts[:len(parts)-1]
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return strings.Join(parts, sep)
}

// ----------------------------------------------------------- stash/sweep --

// restoreStash merges fields hidden under stash[version] back into the
// object (only filling absent paths), coercing values to their declared
// types, then removes that annotation. The type map is the union of every
// version's fields because hidden paths are unknown to the current version
// by definition.
func restoreStash(k *schema.Kind, version string, obj *types.Object) {
	if obj.Metadata.Annotations == nil {
		return
	}
	key := types.StashKey(version)
	raw, ok := obj.Metadata.Annotations[key]
	if !ok {
		return
	}
	var hidden map[string]any
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&hidden); err != nil {
		// An annotation we cannot parse must not be silently discarded;
		// keep it so a human can intervene.
		return
	}
	globalTypes, elementTypes := globalFieldTypeMaps(k)
	for _, section := range []string{"spec", "status"} {
		if sub, ok := hidden[section].(map[string]any); ok {
			coerceTree(sub, section, globalTypes, elementTypes)
			target := sectionMap(obj, section)
			mergeAbsent(target, sub)
		}
	}
	delete(obj.Metadata.Annotations, key)
	if len(obj.Metadata.Annotations) == 0 {
		obj.Metadata.Annotations = nil
	}
}

// pruneAbsorbedStash removes a freshly written stash annotation when every
// value it holds is one the conversion edges can deterministically rebuild
// on the next upgrade: a renamed field, a split source, a wrapped list
// (whose scalar now lives in the object), or an add-rule default. Anything
// else represents genuinely unknown user data and is kept. This is rule-
// based rather than value-based, so equal-valued unrelated fields can never
// cause a false drop.
func pruneAbsorbedStash(k *schema.Kind, version string, obj *types.Object) {
	if obj.Metadata.Annotations == nil {
		return
	}
	key := types.StashKey(version)
	raw, ok := obj.Metadata.Annotations[key]
	if !ok {
		return
	}
	var hidden map[string]any
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&hidden); err != nil {
		return
	}
	droppable := reversiblePaths(k, version)
	// add-rule paths are droppable only while the value still equals the
	// rule default; a user-modified value must stay hidden to round-trip.
	addDefaultsForEdge := addDefaults(k, version)
	// wrap-list sinks are only reversible while the list holds a single
	// element: [v] collapses back to scalar v, but [v1, v2, ...] carries
	// data no old-version path can represent and must stay hidden.
	wrapMultiElement := map[string]bool{}
	for _, r := range k.Rules[version] {
		if r.Kind == schema.RuleWrapList {
			wrapMultiElement[r.Sink] = true
		}
	}
	for _, section := range []string{"spec", "status"} {
		sub, ok := hidden[section].(map[string]any)
		if !ok {
			continue
		}
		for leafPath, leafVal := range flattenLeaves(sub, section) {
			if !droppable[leafPath] {
				return // genuinely unknown user data: keep the stash
			}
			if def, isAdd := addDefaultsForEdge[leafPath]; isAdd {
				if !valuesEqual(leafVal, def) {
					return // user changed the default: preserve it
				}
			}
			if wrapMultiElement[leafPath] {
				if list, isList := leafVal.([]any); !isList || len(list) > 1 {
					return // extra elements would be lost: keep the stash
				}
			}
		}
	}
	delete(obj.Metadata.Annotations, key)
	if len(obj.Metadata.Annotations) == 0 {
		obj.Metadata.Annotations = nil
	}
}

// reversiblePaths lists hidden paths that an upgrade away from version can
// reconstruct without the annotation: new-version names of rename/split
// edges leaving version, wrap-list sinks, and add-rule defaults.
func reversiblePaths(k *schema.Kind, version string) map[string]bool {
	out := map[string]bool{}
	for _, r := range k.Rules[version] {
		switch r.Kind {
		case schema.RuleRename:
			out[r.Sink] = true
		case schema.RuleWrapList:
			out[r.Sink] = true
		case schema.RuleSplit:
			for _, pair := range r.Sources {
				out[pair[1]] = true
			}
		case schema.RuleAdd:
			out[r.Sink] = true
		}
	}
	return out
}

// flattenLeaves returns every leaf path (full dotted path) and its value.
func flattenLeaves(node map[string]any, prefix string) map[string]any {
	out := map[string]any{}
	var walk func(map[string]any, string)
	walk = func(n map[string]any, p string) {
		for k, v := range n {
			path := joinPath(p, k)
			if child, ok := v.(map[string]any); ok {
				walk(child, path)
				continue
			}
			out[path] = v
		}
	}
	walk(node, prefix)
	return out
}

func readLeaf(root map[string]any, path string) (any, bool) {
	segs := splitDotted(path)
	node := root
	for _, seg := range segs[1 : len(segs)-1] {
		next, ok := node[seg].(map[string]any)
		if !ok {
			return nil, false
		}
		node = next
	}
	v, ok := node[segs[len(segs)-1]]
	return v, ok
}

func leafEqual(a, b any) bool {
	// json.Number vs int64 compare on their numeric string form.
	return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}

// valuesEqual compares two leaf values, including lists element-wise.
func valuesEqual(a, b any) bool {
	la, laOK := a.([]any)
	lb, lbOK := b.([]any)
	if laOK || lbOK {
		if !laOK || !lbOK || len(la) != len(lb) {
			return false
		}
		for i := range la {
			if !leafEqual(la[i], lb[i]) {
				return false
			}
		}
		return true
	}
	return leafEqual(a, b)
}

// fieldTypeMap maps every known leaf path of a version to its declared type.
func fieldTypeMap(k *schema.Kind, version string) map[string]schema.FieldType {
	vs, err := k.Version(version)
	if err != nil {
		return map[string]schema.FieldType{}
	}
	out := make(map[string]schema.FieldType, len(vs.Fields))
	for _, f := range vs.Fields {
		out[f.Path] = f.Type
	}
	return out
}

// globalFieldTypeMaps unions field types across all versions. A hidden field
// always has a type in the version that introduced it, even though the
// current version does not know the path. The second map records element
// types of list fields so stash round-trips preserve float/int elements.
func globalFieldTypeMaps(k *schema.Kind) (map[string]schema.FieldType, map[string]schema.FieldType) {
	leaf := map[string]schema.FieldType{}
	elem := map[string]schema.FieldType{}
	for _, vs := range k.Versions {
		for _, f := range vs.Fields {
			leaf[f.Path] = f.Type
			if f.Type == schema.TypeList {
				elem[f.Path] = listElementType(k, f.Path)
			}
		}
	}
	return leaf, elem
}

// listElementType infers a list's element type from wrap-list rules: a list
// produced by RuleWrapList inherits the source scalar's type. Lists created
// directly by the simulator hold strings.
func listElementType(k *schema.Kind, listPath string) schema.FieldType {
	for _, edge := range k.Rules {
		for _, r := range edge {
			if r.Kind == schema.RuleWrapList && r.Sink == listPath {
				if ft, ok := findFieldType(k, r.Source); ok {
					return ft
				}
			}
		}
	}
	return schema.TypeString
}

func findFieldType(k *schema.Kind, path string) (schema.FieldType, bool) {
	for _, vs := range k.Versions {
		for _, f := range vs.Fields {
			if f.Path == path {
				return f.Type, true
			}
		}
	}
	return "", false
}

// addDefaults returns synthesized default values for fields introduced by
// add-rules on the edge that ENTERS fromVersion. When sweeping while
// downgrading away from fromVersion, those values are regenerated on the
// next upgrade, so they do not need to be hidden in an annotation.
func addDefaults(k *schema.Kind, fromVersion string) map[string]any {
	out := map[string]any{}
	for _, r := range k.Rules[fromVersion] {
		if r.Kind == schema.RuleAdd {
			out[r.Sink] = r.Default
		}
	}
	return out
}

func fieldSetOf(k *schema.Kind, version string) map[string]bool {
	set, err := k.FieldSet(version)
	if err != nil {
		return map[string]bool{}
	}
	return set
}

// coerceTree rewrites JSON-flattened leaf values (json.Number/string) back
// to the Go types the schema expects. prefix is the full dotted path, so
// nested leaves are looked up directly in the version's field-type map.
func coerceTree(node map[string]any, prefix string, types, elementTypes map[string]schema.FieldType) {
	for key, val := range node {
		path := joinPath(prefix, key)
		switch tv := val.(type) {
		case map[string]any:
			coerceTree(tv, path, types, elementTypes)
		case []any:
			// Lists only hold scalar elements in these schemas; look up the
			// element type from the field definition and coerce each element
			// (json.Number from an annotation) to that type.
			if want, known := types[path]; known && want == schema.TypeList {
				node[key] = coerceList(tv, elementTypes[path])
			}
		case json.Number:
			// json.Number is a defined string type: it does NOT match a
			// "case string" type switch. Numeric fields coerce to int64/
			// float64; string-typed fields must unwrap to a plain string
			// so the value's Go type matches what the schema declares.
			want, known := types[path]
			if !known || want == schema.TypeString {
				node[key] = tv.String()
			} else {
				node[key] = coerceScalar(tv, want)
			}
		case string:
			// Genuine string. Only reinterpret as a number when the target
			// field is genuinely non-string.
			if want, known := types[path]; known && want != schema.TypeString {
				node[key] = coerceScalar(json.Number(tv), want)
			}
		default:
			// bool, int64, float64 values that reached the stash through
			// an in-memory copy already have the right Go type and need
			// no coercion.
		}
	}
}

func coerceList(list []any, element schema.FieldType) []any {
	for i, v := range list {
		if n, ok := v.(json.Number); ok {
			list[i] = coerceScalar(n, element)
		}
	}
	return list
}

// coerceScalar turns a JSON-stored value into the Go type of the target
// field. Values that do not parse — or fields whose target type is unknown
// (should not happen for schema fields) — keep their raw form rather than
// being mangled: losslessness beats strict typing for hidden data.
func coerceScalar(raw json.Number, want schema.FieldType) any {
	switch want {
	case schema.TypeInt:
		if i, err := raw.Int64(); err == nil {
			return i
		}
		return raw.String()
	case schema.TypeFloat:
		if f, err := raw.Float64(); err == nil {
			return f
		}
		return raw.String()
	case schema.TypeBool:
		if b, err := strconv.ParseBool(raw.String()); err == nil {
			return b
		}
		return raw.String()
	default:
		// Unknown/string-typed field: return the original json.Number,
		// which marshals back identically. Turning it into a plain string
		// would change int fields into strings on a stash round-trip.
		return raw
	}
}

// sweepIntoStash collects every leaf path unknown to targetVersion and
// hides it under stash[targetVersion]. Leaves that merely hold a value
// synthesized by an add-rule entering fromVersion are dropped instead: the
// rule regenerates the identical default on the next upgrade, so hiding it
// would leave a pointless annotation that breaks clean round-trips.
func sweepIntoStash(k *schema.Kind, fromVersion, targetVersion string, obj *types.Object) error {
	known, err := k.FieldSet(targetVersion)
	if err != nil {
		return err
	}
	defaults := addDefaults(k, fromVersion)
	hidden := map[string]any{}
	for _, section := range []string{"spec", "status"} {
		root := sectionMap(obj, section)
		unknown := map[string]any{}
		sweepNode(root, section, known, defaults, unknown)
		if len(unknown) > 0 {
			hidden[section] = unknown
		}
	}
	if len(hidden) == 0 {
		return nil
	}
	raw, err := json.Marshal(hidden)
	if err != nil {
		return err
	}
	if obj.Metadata.Annotations == nil {
		obj.Metadata.Annotations = map[string]string{}
	}
	obj.Metadata.Annotations[types.StashKey(targetVersion)] = string(raw)
	return nil
}

// sweepNode walks a section map and moves unknown leaves into unknown.
// Known container maps are recursed into; unknown subtrees move wholesale.
// A map node is only treated as a known container when the schema actually
// has a leaf beneath that path — otherwise an unknown nested object that
// happens to share a section name (e.g. a user field named "spec") would be
// recursed into and leave an empty shell behind.
// An unknown leaf equal to an add-rule default is discarded rather than
// hidden (the default is regenerated deterministically on re-upgrade).
func sweepNode(node map[string]any, prefix string, known map[string]bool, defaults map[string]any, unknown map[string]any) {
	for key, val := range node {
		path := joinPath(prefix, key)
		if known[path] {
			if child, ok := val.(map[string]any); ok && hasKnownLeafUnder(known, path) {
				sweepNode(child, path, known, defaults, unknown)
				continue
			}
			if _, isMap := val.(map[string]any); isMap && !hasKnownLeafUnder(known, path) {
				// known as a bare container name but no schema leaf lives
				// under it: hide the whole subtree
				setNested(unknown, path, val)
				delete(node, key)
				continue
			}
			continue
		}
		if def, isDefault := defaults[path]; isDefault && valuesEqual(val, def) {
			delete(node, key)
			continue
		}
		setNested(unknown, path, val)
		delete(node, key)
	}
}

// hasKnownLeafUnder reports whether the field set contains a leaf path
// beneath prefix (prefix itself being a section/container does not count).
func hasKnownLeafUnder(known map[string]bool, prefix string) bool {
	stem := prefix + "."
	for p := range known {
		if len(p) > len(stem) && p[:len(stem)] == stem {
			return true
		}
	}
	return false
}

// mergeAbsent deep-merges src into dst without overwriting existing leaves.
func mergeAbsent(dst, src map[string]any) {
	for k, v := range src {
		existing, ok := dst[k]
		if !ok {
			dst[k] = deepCopy(v)
			continue
		}
		if em, eok := existing.(map[string]any); eok {
			if sm, sok := v.(map[string]any); sok {
				mergeAbsent(em, sm)
			}
		}
	}
}

// ------------------------------------------------------------- path utils --

func sectionMap(obj *types.Object, section string) map[string]any {
	switch section {
	case "spec":
		if obj.Spec == nil {
			obj.Spec = map[string]any{}
		}
		return obj.Spec
	case "status":
		if obj.Status == nil {
			obj.Status = map[string]any{}
		}
		return obj.Status
	default:
		panic("unknown section " + section)
	}
}

func splitDotted(p string) []string {
	return strings.Split(p, ".")
}

func joinPath(prefix, key string) string {
	if prefix == "" {
		return key
	}
	return prefix + "." + key
}

// getPath resolves a dotted path like "spec.replica".
func getPath(obj *types.Object, path string) (any, bool) {
	segs := splitDotted(path)
	var node map[string]any = sectionMap(obj, segs[0])
	for _, seg := range segs[1 : len(segs)-1] {
		next, ok := node[seg].(map[string]any)
		if !ok {
			return nil, false
		}
		node = next
	}
	v, ok := node[segs[len(segs)-1]]
	return v, ok
}

func setPath(obj *types.Object, path string, val any) {
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

func deletePath(obj *types.Object, path string) {
	segs := splitDotted(path)
	node := sectionMap(obj, segs[0])
	for _, seg := range segs[1 : len(segs)-1] {
		next, ok := node[seg].(map[string]any)
		if !ok {
			return
		}
		node = next
	}
	delete(node, segs[len(segs)-1])
}

// setNested places val at a dotted path (e.g. "spec.replicas", possibly
// deeper) inside the stash subtree. root is already the section's stash
// container, so the leading section segment is skipped.
func setNested(root map[string]any, path string, val any) {
	segs := splitDotted(path)
	if len(segs) > 0 && (segs[0] == "spec" || segs[0] == "status") {
		segs = segs[1:]
	}
	node := root
	for _, seg := range segs[:len(segs)-1] {
		next, ok := node[seg].(map[string]any)
		if !ok {
			next = map[string]any{}
			node[seg] = next
		}
		node = next
	}
	node[segs[len(segs)-1]] = deepCopy(val)
}

func deepCopy(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = deepCopy(val)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i := range t {
			out[i] = deepCopy(t[i])
		}
		return out
	default:
		return v
	}
}
