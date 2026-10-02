"""
content JSON 容错回归测试（P2-6）。

背景：各层 schema 都是 additionalProperties: False，content JSON 里多写一个键
就会整份文件校验失败并被静默跳过 —— 用户在界面上只看到「这个学科凭空消失」。
修复：校验前按 schema 白名单剔除非定义字段，并记录被剔除的键名。

    python tests/test_content_tolerance.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.data_manager.content_manager import (  # noqa: E402
    CONTENT_SCHEMA,
    sanitize_content,
)

try:
    from jsonschema import validate
    HAVE_JSONSCHEMA = True
except ImportError:  # pragma: no cover
    HAVE_JSONSCHEMA = False


def _valid_content():
    return {
        "subject": "Science",
        "grade": 10,
        "topics": [{
            "name": "第一章",
            "subtopics": [{
                "name": "第一节",
                "concepts": [{
                    "name": "概念A",
                    "summary": "摘要",
                    "steps": ["步骤1"],
                    "questions": [{
                        "question": "问题？",
                        "acceptable_answers": ["答案"],
                        "hints": ["提示"],
                    }],
                }],
            }],
        }],
    }


def test_extra_keys_are_stripped_not_fatal():
    """核心回归：多写字段不应导致整份 content 被丢弃。"""
    c = _valid_content()
    c["note"] = "老师的备注"                     # 顶层多余键
    c["topics"][0]["order"] = 1                  # topic 层多余键
    c["topics"][0]["subtopics"][0]["difficulty"] = "中"
    c["topics"][0]["subtopics"][0]["concepts"][0]["source_page"] = 12
    c["topics"][0]["subtopics"][0]["concepts"][0]["questions"][0]["score"] = 5

    dropped = []
    clean = sanitize_content(c, dropped)

    assert "note" not in clean
    assert "order" not in clean["topics"][0]
    assert "difficulty" not in clean["topics"][0]["subtopics"][0]
    assert "source_page" not in clean["topics"][0]["subtopics"][0]["concepts"][0]
    assert "score" not in clean["topics"][0]["subtopics"][0]["concepts"][0]["questions"][0]

    # 业务字段完好
    assert clean["subject"] == "Science"
    assert clean["topics"][0]["subtopics"][0]["concepts"][0]["name"] == "概念A"

    # 记录到被剔除清单，便于提示老师
    assert any("note" in d for d in dropped), dropped

    if HAVE_JSONSCHEMA:
        validate(instance=clean, schema=CONTENT_SCHEMA)  # 清洗后可过严 schema


def test_missing_required_field_still_fails():
    """容错不等于放水：真正的结构错误仍必须报错。"""
    c = _valid_content()
    del c["grade"]
    clean = sanitize_content(c, [])
    assert "grade" not in clean
    if HAVE_JSONSCHEMA:
        try:
            validate(instance=clean, schema=CONTENT_SCHEMA)
            raised = False
        except Exception:  # noqa: BLE001
            raised = True
        assert raised, "缺少必填字段竟然通过了校验"


def test_wrong_type_still_fails():
    c = _valid_content()
    c["grade"] = 99          # 超出 1–12
    clean = sanitize_content(c, [])
    if HAVE_JSONSCHEMA:
        try:
            validate(instance=clean, schema=CONTENT_SCHEMA)
            raised = False
        except Exception:  # noqa: BLE001
            raised = True
        assert raised, "非法年级竟然通过了校验"


def test_sanitize_does_not_mutate_input():
    c = _valid_content()
    c["note"] = "x"
    clean = sanitize_content(c, [])
    assert "note" in c, "原始对象被就地修改了"
    assert clean is not c


def test_sanitize_handles_malformed_shapes():
    """畸形数据不应抛异常（交由 schema 报错）。"""
    for bad in [None, [], "str", 123, {"subject": "S", "grade": 10, "topics": "not-a-list"}]:
        out = sanitize_content(bad, [])
        assert out is not None or bad is None


def test_deeply_nested_subtopics_are_guarded():
    """超深嵌套要有保护，避免递归爆栈。"""
    node = {"name": "L0", "subtopics": []}
    cur = node
    for i in range(12):
        nxt = {"name": f"L{i+1}", "subtopics": []}
        cur["subtopics"].append(nxt)
        cur = nxt
    c = {"subject": "S", "grade": 10, "topics": [{"name": "T", "subtopics": [node]}]}
    clean = sanitize_content(c, [])   # 不抛异常即可
    assert clean["topics"][0]["subtopics"][0]["name"] == "L0"


def _main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, t in tests:
        try:
            t()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
