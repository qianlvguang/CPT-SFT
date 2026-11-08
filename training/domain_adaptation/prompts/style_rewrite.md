# 严谨详尽、带依据引用的风格改写提示词

只接收已通过 QA 核验的候选；改写后再次经过同一套事实和引用检查。

```text
将回答改写为专业、严谨详尽、带依据引用的中文回答。

必须保持原答案的事实、结论、数值、日期、适用条件、例外和不确定性。
只能使用给定来源；不得新增未验证的结论，不得编造引用 key、URL 或法条。
先明确回答，再解释必要依据和推导；复杂问题可分节，简单问题自然用段落。
引用贴近被支持的主张。把“来源明确说明”“根据前提推导”“材料不足”区分清楚。
删去空洞奉承、重复结论和无关免责声明，但不要删去影响正确性的限定条件。
如原答案本身不能被来源支持，返回 needs_review，不尝试凭空修成正确答案。

仅返回 JSON：
{
  "status": "rewritten|needs_review",
  "answer": "改写后的最终回答",
  "citations": ["S1"],
  "changed_factual_content": false,
  "review_reason": ""
}

source_bundle:
{{source_bundle}}
question:
{{question}}
verified_answer:
{{verified_answer}}
```

`changed_factual_content=false` 是生成器声明，不能作为事实保持的证明。正式数据仍需通过验证器与引用程序核验。
