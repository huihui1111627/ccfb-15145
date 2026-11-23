// Package web serves the operator UI: kind/version inventory, live
// migration progress, per-object multi-version views and fault controls.
package web

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"time"

	"crv/pkg/convert"
	"crv/pkg/schema"
	"crv/pkg/simulator"
)

// Server holds the simulated platform and its HTTP handlers.
type Server struct {
	platform *simulator.Platform
	mux      *http.ServeMux
}

// Serve builds a platform, seeds it and starts the HTTP server.
func Serve(addr string, seed int64, objectsPerKind int) error {
	p, err := simulator.NewPlatform(seed)
	if err != nil {
		return err
	}
	if err := p.SeedObjects(objectsPerKind); err != nil {
		return err
	}
	s := &Server{platform: p, mux: http.NewServeMux()}
	s.routes()
	fmt.Printf("CRV simulator UI listening on %s (seed=%d, %d kinds)\n",
		addr, seed, len(p.Registry.All()))
	return http.ListenAndServe(addr, s.mux)
}

func (s *Server) routes() {
	s.mux.HandleFunc("/", s.handleIndex)
	s.mux.HandleFunc("/api/kinds", s.handleKinds)
	s.mux.HandleFunc("/api/kinds/", s.handleKindDetail)
	s.mux.HandleFunc("/api/object", s.handleObject)
	s.mux.HandleFunc("/api/migrations", s.handleMigrations)
	s.mux.HandleFunc("/api/migrate", s.handleMigrate)
	s.mux.HandleFunc("/api/faults", s.handleFaults)
	s.mux.HandleFunc("/api/metrics", s.handleMetrics)
}

// KindView is one row of the inventory table.
type KindView struct {
	Group          string         `json:"group"`
	Kind           string         `json:"kind"`
	Versions       []string       `json:"versions"`
	StorageVersion string         `json:"storage_version"`
	StoredByVersion map[string]int `json:"stored_by_version"`
	ObjectCount    int            `json:"object_count"`
	Edges          []EdgeView     `json:"edges"`
}

// EdgeView summarizes one adjacent-version conversion edge.
type EdgeView struct {
	From  string   `json:"from"`
	To    string   `json:"to"`
	Rules []string `json:"rules"`
}

func (s *Server) handleKinds(w http.ResponseWriter, r *http.Request) {
	var views []KindView
	for _, k := range s.platform.Registry.All() {
		views = append(views, s.kindView(k))
	}
	writeJSON(w, views)
}

func (s *Server) kindView(k *schema.Kind) KindView {
	versions := make([]string, len(k.Versions))
	for i, vs := range k.Versions {
		versions[i] = vs.Name
	}
	edges := make([]EdgeView, 0, len(k.Rules))
	for i := 0; i < len(k.Versions)-1; i++ {
		from := k.Versions[i].Name
		var ruleDesc []string
		for _, rule := range k.Rules[from] {
			ruleDesc = append(ruleDesc, describeRule(rule))
		}
		edges = append(edges, EdgeView{
			From: from, To: k.Versions[i+1].Name, Rules: ruleDesc,
		})
	}
	return KindView{
		Group: k.Group, Kind: k.Name, Versions: versions,
		StorageVersion:  k.StorageVersion,
		StoredByVersion: s.platform.Store.CountByStorageVersion(k.Name),
		ObjectCount:     s.platform.Store.Count(k.Name),
		Edges:           edges,
	}
}

func describeRule(rule *schema.ConversionRule) string {
	switch rule.Kind {
	case schema.RuleRename:
		return fmt.Sprintf("rename %s → %s", rule.Source, rule.Sink)
	case schema.RuleAdd:
		return fmt.Sprintf("add %s (default %v)", rule.Sink, rule.Default)
	case schema.RuleSplit:
		return fmt.Sprintf("split %s → %d fields (join %q)", rule.Sources[0][0], len(rule.Sources), rule.MergeSep)
	case schema.RuleWrapList:
		return fmt.Sprintf("wrap %s → list %s", rule.Source, rule.Sink)
	default:
		return string(rule.Kind)
	}
}

// ObjectView shows one object rendered in every served version, plus its
// raw stored envelope.
type ObjectView struct {
	Kind           string                 `json:"kind"`
	Key            string                 `json:"key"`
	StoredVersion  string                 `json:"stored_version"`
	VersionViews   map[string]json.RawMessage `json:"version_views"`
	StoredEnvelope json.RawMessage        `json:"stored_envelope"`
}

func (s *Server) handleKindDetail(w http.ResponseWriter, r *http.Request) {
	// /api/kinds/{name}/objects
	kind := r.URL.Path[len("/api/kinds/"):]
	// strip the trailing /objects segment if present
	for i := len(kind) - 1; i >= 0; i-- {
		if kind[i] == '/' {
			kind = kind[:i]
			break
		}
	}
	k, err := s.platform.Registry.Get(kind)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	objs, err := s.platform.Store.List(kind, k.StorageVersion)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	keys := make([]string, len(objs))
	for i, o := range objs {
		keys[i] = o.Key()
	}
	sort.Strings(keys)
	writeJSON(w, keys)
}

func (s *Server) handleObject(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	kind, ns, name := q.Get("kind"), q.Get("namespace"), q.Get("name")
	if kind == "" || name == "" {
		http.Error(w, "kind and name are required", http.StatusBadRequest)
		return
	}
	k, err := s.platform.Registry.Get(kind)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	storedVersion, raw, ok := s.platform.Store.RawRecord(kind, ns, name)
	if !ok {
		http.Error(w, "object not found", http.StatusNotFound)
		return
	}
	envelope, _ := json.Marshal(raw)
	view := ObjectView{
		Kind: kind, Key: ns + "/" + name, StoredVersion: storedVersion,
		VersionViews:   map[string]json.RawMessage{},
		StoredEnvelope: envelope,
	}
	for _, vs := range k.Versions {
		obj, err := s.platform.Store.Get(kind, ns, name, vs.Name)
		if err != nil {
			writeJSON(w, map[string]string{"error": err.Error()})
			return
		}
		data, _ := json.MarshalIndent(obj, "", "  ")
		view.VersionViews[vs.Name] = data
	}
	writeJSON(w, view)
}

func (s *Server) handleMigrations(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.platform.Migrator.AllProgress())
}

func (s *Server) handleMigrate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Kind    string `json:"kind"`
		Version string `json:"version"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	prog, err := s.platform.Migrator.SwitchStorage(req.Kind, req.Version)
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	writeJSON(w, prog)
}

func (s *Server) handleFaults(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, s.platform.Service.Faults())
	case http.MethodPost:
		var req struct {
			Kind string          `json:"kind"`
			Fault convert.Fault `json:"fault"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		s.platform.Service.SetFault(req.Kind, req.Fault)
		writeJSON(w, map[string]string{"status": "applied"})
	case http.MethodDelete:
		s.platform.Service.ClearFaults()
		writeJSON(w, map[string]string{"status": "cleared"})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	rows, total, failed := s.platform.Service.Report()
	writeJSON(w, map[string]any{
		"rows": rows, "total_calls": total, "failed_calls": failed,
		"server_time": time.Now().Format(time.RFC3339),
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}
