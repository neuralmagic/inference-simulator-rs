// Command kv-events-smoke is a live interop check: it subscribes to a running KV-cache
// event publisher and decodes the messages with the *real* llm-d-kv-cache consumer
// (`engineadapter.VLLMAdapter.ParseMessage`), the exact code path the router's KV-events
// indexer uses. If it decodes our bytes into the expected BlockStored / BlockRemoved /
// AllBlocksCleared events, our wire format is router-compatible.
//
// Run it against the Rust emitter example:
//
//	cargo run --example kv_event_emitter -- 'tcp://*:5556' 'kv@127.0.0.1:8000@mock-model'
//	KV_ENDPOINT=tcp://127.0.0.1:5556 go run .
//
// or via scripts/kv_events_smoke.sh, which wires both ends up and tears them down.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "FAIL: "+format+"\n", args...)
	os.Exit(1)
}

func main() {
	endpoint := getenv("KV_ENDPOINT", "tcp://127.0.0.1:5556")
	topicFilter := getenv("KV_TOPIC_FILTER", "kv@")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	// Subscribe exactly as the router does: a SUB socket dialing the pod's PUB endpoint,
	// filtered on the `kv@` topic prefix.
	sub := zmq4.NewSub(ctx)
	defer sub.Close()
	if err := sub.Dial(endpoint); err != nil {
		fatalf("dial %s: %v", endpoint, err)
	}
	if err := sub.SetOption(zmq4.OptionSubscribe, topicFilter); err != nil {
		fatalf("subscribe %q: %v", topicFilter, err)
	}

	adapter, err := engineadapter.NewAdapter("vllm")
	if err != nil {
		fatalf("new vllm adapter: %v", err)
	}

	fmt.Printf("subscribed to %s (filter %q); waiting for events...\n", endpoint, topicFilter)

	var sawStored, sawRemoved, sawCleared bool
	for !(sawStored && sawRemoved && sawCleared) {
		if ctx.Err() != nil {
			fatalf("timed out before observing all event types (stored=%v removed=%v cleared=%v)",
				sawStored, sawRemoved, sawCleared)
		}
		msg, err := sub.Recv()
		if err != nil {
			fatalf("recv: %v", err)
		}
		if len(msg.Frames) != 3 {
			fatalf("expected 3 frames (topic, seq, payload), got %d", len(msg.Frames))
		}

		raw := &kvevents.RawMessage{
			Topic:    string(msg.Frames[0]),
			Sequence: binary.BigEndian.Uint64(msg.Frames[1]),
			Payload:  msg.Frames[2],
		}

		// The real consumer decode path.
		podID, model, batch, err := adapter.ParseMessage(raw)
		if err != nil {
			fatalf("real llm-d-kv-cache decoder rejected our bytes: %v", err)
		}
		fmt.Printf("decoded batch: pod=%q model=%q events=%d seq=%d\n",
			podID, model, len(batch.Events), raw.Sequence)

		for _, e := range batch.Events {
			switch ev := e.(type) {
			case *kvevents.BlockStoredEvent:
				fmt.Printf("  BlockStored hashes=%v tokens=%d parent=%d tier=%q\n",
					ev.BlockHashes, len(ev.Tokens), ev.ParentHash, ev.DeviceTier)
				sawStored = true
			case *kvevents.BlockRemovedEvent:
				fmt.Printf("  BlockRemoved hashes=%v tier=%q\n", ev.BlockHashes, ev.DeviceTier)
				sawRemoved = true
			case *kvevents.AllBlocksClearedEvent:
				fmt.Printf("  AllBlocksCleared\n")
				sawCleared = true
			default:
				fatalf("unexpected event type %T", e)
			}
		}
	}

	fmt.Println("PASS: real llm-d-kv-cache decoder accepted BlockStored, BlockRemoved, and AllBlocksCleared")
}
