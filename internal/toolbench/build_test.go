package toolbench

import "testing"

func TestReplyText(t *testing.T) {
	for in, want := range map[string]string{
		`<TOOLCALL>[{"name": "a", "arguments": {"x": 1}}]</TOOLCALL>`:                           `{"name": "a", "arguments": {"x": 1}}`,
		`<TOOLCALL>[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]</TOOLCALL>`: "{\"name\": \"a\", \"arguments\": {}}\n{\"name\": \"b\", \"arguments\": {}}",
		"Which city do you mean?":       "Which city do you mean?",
		"<TOOLCALL>not json</TOOLCALL>": "<TOOLCALL>not json</TOOLCALL>",
	} {
		if got := replyText(in); got != want {
			t.Errorf("replyText(%q) = %q, want %q", in, got, want)
		}
	}
}
