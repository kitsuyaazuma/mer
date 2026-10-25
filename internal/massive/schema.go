package massive

import (
	"fmt"
	"strings"

	"github.com/gomlx/go-huggingface/tokenizers"

	"github.com/kitsuyaazuma/mer/internal/tokspan"
)

// Option is one possible decision.
type Option struct {
	ID   string // machine name (the MASSIVE label)
	Desc string // short natural-language description
}

// Schema is the question and the (canonical order) list of options.
type Schema struct {
	Name     string
	Question string
	Options  []Option
	Answer   string // appended after the options as an empty answer slot (e.g., "\nANSWER:")
}

const question = "ユーザーの発話の意図はどれですか？"

// Index returns the canonical index of the option with the given ID, or -1.
func (s Schema) Index(id string) int {
	for i, o := range s.Options {
		if o.ID == id {
			return i
		}
	}
	return -1
}

// Scenarios10 is a 10-way schema over 10 of the 18 MASSIVE scenarios.
func Scenarios10() Schema {
	return Schema{Name: "scenario10", Question: question, Options: []Option{
		{"weather", "天気や気温、天気予報について知りたい"},
		{"alarm", "アラームを設定・確認・削除したい"},
		{"music", "音楽の好き嫌いを伝えたり、曲について尋ねたり、再生を操作したりしたい"},
		{"calendar", "カレンダーの予定を登録・確認・削除したい"},
		{"news", "最新のニュースを知りたい"},
		{"transport", "電車や渋滞など交通について調べたり、タクシーや切符を予約したりしたい"},
		{"takeaway", "テイクアウトや出前を注文・確認したい"},
		{"email", "メールの送信・確認や連絡先の管理をしたい"},
		{"social", "SNSへの投稿や確認をしたい"},
		{"lists", "リストやToDoを作成・確認・削除したい"},
	}}
}

// Intents60 is the full MASSIVE intent inventory (60-way).
func Intents60() Schema {
	return Schema{Name: "intent60", Question: question, Options: []Option{
		{"alarm_query", "設定済みのアラームを確認したい"},
		{"alarm_remove", "アラームを削除・解除したい"},
		{"alarm_set", "アラームを設定したい"},
		{"audio_volume_down", "音量を下げたい"},
		{"audio_volume_mute", "音を消したい（ミュート）"},
		{"audio_volume_other", "音量を指定の大きさに変えたい"},
		{"audio_volume_up", "音量を上げたい"},
		{"calendar_query", "カレンダーの予定を確認したい"},
		{"calendar_remove", "カレンダーの予定を削除したい"},
		{"calendar_set", "カレンダーに予定を登録したい"},
		{"cooking_query", "料理に関する質問をしたい"},
		{"cooking_recipe", "料理のレシピを知りたい"},
		{"datetime_convert", "時刻を別のタイムゾーンに変換したい"},
		{"datetime_query", "今の日付や時刻を知りたい"},
		{"email_addcontact", "連絡先を追加したい"},
		{"email_query", "メールを確認したい"},
		{"email_querycontact", "連絡先の情報を調べたい"},
		{"email_sendemail", "メールを送信したい"},
		{"general_greet", "挨拶をしたい"},
		{"general_joke", "冗談を言ってほしい"},
		{"general_quirky", "雑談や変わった質問をしたい"},
		{"iot_cleaning", "掃除ロボットを操作したい"},
		{"iot_coffee", "コーヒーメーカーを操作したい"},
		{"iot_hue_lightchange", "照明の色を変えたい"},
		{"iot_hue_lightdim", "照明を暗くしたい"},
		{"iot_hue_lightoff", "照明を消したい"},
		{"iot_hue_lighton", "照明をつけたい"},
		{"iot_hue_lightup", "照明を明るくしたい"},
		{"iot_wemo_off", "スマートプラグをオフにしたい"},
		{"iot_wemo_on", "スマートプラグをオンにしたい"},
		{"lists_createoradd", "リストを作成・項目を追加したい"},
		{"lists_query", "リストの内容を確認したい"},
		{"lists_remove", "リストから項目を削除したい"},
		{"music_dislikeness", "曲が好きではないと伝えたい"},
		{"music_likeness", "曲が好きだと伝えたい"},
		{"music_query", "流れている曲について尋ねたい"},
		{"music_settings", "音楽の再生設定を変えたい"},
		{"news_query", "ニュースを知りたい"},
		{"play_audiobook", "オーディオブックを再生したい"},
		{"play_game", "ゲームをしたい"},
		{"play_music", "音楽を再生したい"},
		{"play_podcasts", "ポッドキャストを再生したい"},
		{"play_radio", "ラジオを再生したい"},
		{"qa_currency", "為替レートや通貨換算を知りたい"},
		{"qa_definition", "言葉の意味を調べたい"},
		{"qa_factoid", "事実や一般知識について質問したい"},
		{"qa_maths", "計算や数学の質問をしたい"},
		{"qa_stock", "株価を知りたい"},
		{"recommendation_events", "おすすめのイベントを知りたい"},
		{"recommendation_locations", "おすすめの場所やお店を知りたい"},
		{"recommendation_movies", "おすすめの映画を知りたい"},
		{"social_post", "SNSに投稿したい"},
		{"social_query", "SNSの投稿や通知を確認したい"},
		{"takeaway_order", "食べ物を注文したい"},
		{"takeaway_query", "注文した食べ物の状況を確認したい"},
		{"transport_query", "交通手段や経路・時刻を調べたい"},
		{"transport_taxi", "タクシーを呼びたい"},
		{"transport_ticket", "乗車券を購入・予約したい"},
		{"transport_traffic", "渋滞や交通状況を知りたい"},
		{"weather_query", "天気や気温を知りたい"},
	}}
}

// Prompt is the decision prompt of one utterance, tokenized as a single sequence, with the position of every option.
type Prompt struct {
	SrcID string
	Text  string // the prompt as text

	IDs       []int32    // token IDs, including <bos>
	OptSpans  [][2]int32 // [start,end) tokens of the option shown at position j
	StateSpan [2]int32   // [start,end) tokens of the utterance

	Order []int // Order[j] = canonical option index shown at position j
	Gold  int   // position j of the correct option (the training label)
	Canon int   // canonical index of the correct option
}

// BuildPrompt writes the prompt for one utterance and tokenizes it.
func BuildPrompt(tok tokenizers.Tokenizer, s Schema, order []int, utt string, canon int) (Prompt, error) {
	var sb strings.Builder
	// The state comes first because Gemma is causal: every option token must have read it.
	sb.WriteString("STATE:\n")
	stateLo := sb.Len()
	sb.WriteString(utt)
	stateHi := sb.Len()
	sb.WriteString("\nQUESTION:\n")
	sb.WriteString(s.Question)
	sb.WriteString("\nOPTIONS:\n")
	type byteRange struct{ lo, hi int }
	optRanges := make([]byteRange, len(order))
	gold := -1
	for j, c := range order {
		if j > 0 {
			sb.WriteString("\n")
		}
		lo := sb.Len()
		fmt.Fprintf(&sb, "%s: %s", s.Options[c].ID, s.Options[c].Desc)
		optRanges[j] = byteRange{lo, sb.Len()}
		if c == canon {
			gold = j
		}
	}
	if gold < 0 {
		return Prompt{}, fmt.Errorf("gold option %d not among the presented options", canon)
	}
	sb.WriteString(s.Answer)
	text := sb.String()

	ranges := make([][2]int, 1+len(optRanges))
	ranges[0] = [2]int{stateLo, stateHi}
	for j, r := range optRanges {
		ranges[j+1] = [2]int{r.lo, r.hi}
	}
	ids, spans, err := tokspan.Encode(tok, text, ranges)
	if err != nil {
		return Prompt{}, err
	}
	p := Prompt{Text: text, IDs: ids, Order: order, Gold: gold, Canon: canon}
	p.StateSpan, p.OptSpans = spans[0], spans[1:]
	return p, nil
}
