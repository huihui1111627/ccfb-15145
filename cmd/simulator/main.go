// Command simulator runs the multi-version custom-resource platform
// simulation. Without flags it executes the full acceptance suite; with
// -serve it starts the HTTP UI instead.
package main

import (
	"flag"
	"fmt"
	"os"

	"crv/pkg/acceptance"
	"crv/cmd/simulator/web"
)

func main() {
	var (
		seed        = flag.Int64("seed", 42, "deterministic simulation seed")
		objects     = flag.Int("objects", 5, "objects seeded per kind")
		traffic     = flag.Int("traffic", 3000, "mixed-version traffic operations")
		serve       = flag.Bool("serve", false, "start the web UI instead of running acceptance")
		addr        = flag.String("addr", ":8080", "web UI listen address")
	)
	flag.Parse()

	if *serve {
		if err := web.Serve(*addr, *seed, *objects); err != nil {
			fmt.Fprintln(os.Stderr, "server error:", err)
			os.Exit(1)
		}
		return
	}

	report := acceptance.Run(*seed, *objects, *traffic)
	fmt.Print(report.Pretty())
	if !report.OK() {
		os.Exit(1)
	}
}
