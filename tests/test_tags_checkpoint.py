import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from photowall import assets, tags
from photowall.checkpoints import (
    cleanup_expired_checkpoints,
    load_checkpoints,
)
from photowall.cli import build_parser
from photowall.tags import (
    get_checkpoint_dir,
    save_checkpoint,
)


def _create_dummy_image(path: Path, color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path)


def test_save_and_load_checkpoints(tmp_path):
    checkpoint_dir = get_checkpoint_dir(tmp_path)
    assert checkpoint_dir == tmp_path / ".photowall" / "checkpoints" / "tags"

    # 1. 保存分块 1
    chunk_1 = {
        "img1.jpg": {"size": 100, "mtime": 1000, "raw": {"tag1": 0.9}},
        "img2.jpg": {"size": 200, "mtime": 2000, "raw": {"tag2": 0.8}},
    }
    saved_file_1 = save_checkpoint(checkpoint_dir, chunk_1, "0001")
    assert saved_file_1 is not None
    assert saved_file_1.exists()
    assert saved_file_1.name == "tags_0001.json"

    # 2. 保存分块 2
    chunk_2 = {
        "img3.jpg": {"size": 300, "mtime": 3000, "raw": {"tag3": 0.7}},
    }
    saved_file_2 = save_checkpoint(checkpoint_dir, chunk_2, "0002")
    assert saved_file_2 is not None
    assert saved_file_2.exists()
    assert saved_file_2.name == "tags_0002.json"

    # 空数据不保存
    assert save_checkpoint(checkpoint_dir, {}, "empty") is None

    # 3. 加载并合并
    loaded = load_checkpoints(checkpoint_dir)
    assert len(loaded) == 3
    assert "img1.jpg" in loaded
    assert "img2.jpg" in loaded
    assert "img3.jpg" in loaded
    assert loaded["img1.jpg"]["raw"] == {"tag1": 0.9}
    assert loaded["img3.jpg"]["raw"] == {"tag3": 0.7}


def test_load_checkpoints_latest_wins(tmp_path):
    """验证多个 checkpoints 遇到重复文件名时以时间最新的为准。"""
    checkpoint_dir = get_checkpoint_dir(tmp_path)
    checkpoint_dir.mkdir(parents=True)

    f_old = save_checkpoint(
        checkpoint_dir,
        {"dup.jpg": {"size": 100, "mtime": 1000, "raw": {"old_tag": 0.6}}},
        "zzz_old",
    )
    f_new = save_checkpoint(
        checkpoint_dir,
        {"dup.jpg": {"size": 200, "mtime": 2000, "raw": {"new_tag": 0.95}}},
        "aaa_new",
    )
    assert f_old is not None and f_new is not None

    # 显式设置文件修改时间：让 f_new 比 f_old 更新（即便文件名 aaa_new 字典序靠前）
    now_ts = datetime.now().timestamp()
    os.utime(f_old, (now_ts - 3600, now_ts - 3600))
    os.utime(f_new, (now_ts - 60, now_ts - 60))

    loaded = load_checkpoints(checkpoint_dir)
    assert loaded["dup.jpg"]["size"] == 200
    assert loaded["dup.jpg"]["mtime"] == 2000
    assert loaded["dup.jpg"]["raw"] == {"new_tag": 0.95}


def test_cleanup_expired_checkpoints(tmp_path):
    """验证仅删除超过 7 天的过期 checkpoints 及 .tmp 文件，保留近期 checkpoints。"""
    checkpoint_dir = get_checkpoint_dir(tmp_path)
    checkpoint_dir.mkdir(parents=True)

    recent_file = checkpoint_dir / "tags_recent.json"
    expired_file = checkpoint_dir / "tags_expired.json"
    tmp_file = checkpoint_dir / "tags_temp.json.tmp"
    keep_txt = checkpoint_dir / "keep_me.txt"

    recent_file.write_text("{}", encoding="utf-8")
    expired_file.write_text("{}", encoding="utf-8")
    tmp_file.write_text("{}", encoding="utf-8")
    keep_txt.write_text("keep", encoding="utf-8")

    now_ts = datetime.now().timestamp()
    # 设置为 8 天前
    eight_days_ago = now_ts - 8 * 86400
    os.utime(expired_file, (eight_days_ago, eight_days_ago))

    removed = cleanup_expired_checkpoints(checkpoint_dir)
    assert removed == 2
    assert recent_file.exists()
    assert not expired_file.exists()
    assert not tmp_file.exists()
    assert keep_txt.exists()


def test_load_checkpoints_with_corrupted_file(tmp_path, capsys):
    checkpoint_dir = get_checkpoint_dir(tmp_path)
    checkpoint_dir.mkdir(parents=True)

    # 正常文件
    valid_data = {"img1.jpg": {"size": 100, "mtime": 1000, "raw": {}}}
    (checkpoint_dir / "tags_001.json").write_text(
        json.dumps(valid_data), encoding="utf-8"
    )

    # 损坏文件
    (checkpoint_dir / "tags_corrupted.json").write_text(
        "{ bad_json ...", encoding="utf-8"
    )

    loaded = load_checkpoints(checkpoint_dir)
    assert len(loaded) == 1
    assert "img1.jpg" in loaded

    captured = capsys.readouterr()
    assert "读取断点缓存文件 tags_corrupted.json 失败" in captured.out


def test_cli_checkpoint_interval_args():
    parser = build_parser()
    args_tags = parser.parse_args(["tags", "--checkpoint-interval", "50"])
    assert args_tags.checkpoint_interval == 50

    args_assets = parser.parse_args(["assets", "--checkpoint-interval", "80"])
    assert args_assets.checkpoint_interval == 80

    args_index = parser.parse_args(["index", "--checkpoint-interval", "200"])
    assert args_index.checkpoint_interval == 200


def test_tags_resume_from_checkpoint_and_retention(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    img1 = assets_dir / "pic1.jpg"
    img2 = assets_dir / "pic2.jpg"
    _create_dummy_image(img1)
    _create_dummy_image(img2)

    stat1 = img1.stat()

    # 模拟在中断前，pic1.jpg 已经存在断点缓存中，同时存在一个 8 天前的过期 checkpoint
    checkpoint_dir = get_checkpoint_dir(tmp_path)
    checkpoint_dir.mkdir(parents=True)
    cp_file_1 = checkpoint_dir / "tags_0001.json"
    cp_data = {
        "pic1.jpg": {
            "size": stat1.st_size,
            "mtime": int(stat1.st_mtime),
            "raw": {"cat_ears": 0.95},
        }
    }
    cp_file_1.write_text(json.dumps(cp_data), encoding="utf-8")

    expired_cp = checkpoint_dir / "tags_expired.json"
    expired_cp.write_text(
        json.dumps({"old.jpg": {"size": 1, "mtime": 1, "raw": {}}}), encoding="utf-8"
    )
    old_ts = datetime.now().timestamp() - 8 * 86400
    os.utime(expired_cp, (old_ts, old_ts))

    # Mock 汉化字典、模型下载与推理会话
    monkeypatch.setattr(tags, "load_translations", lambda *a: {})
    monkeypatch.setattr(tags, "get_model_and_tags", lambda _: ("mock.onnx", "mock.csv"))
    monkeypatch.setattr(
        tags, "load_tags_csv", lambda _: (["cat_ears", "blue_hair"], [0, 0])
    )

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 448, 448]
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.array([[2.0, 2.0]])]

    monkeypatch.setattr(
        tags, "create_inference_session", lambda *a, **kw: (mock_session, "CPU")
    )

    args = argparse.Namespace(
        home=tmp_path,
        dir=tmp_path,
        force=False,
        filter=False,
        device="cpu",
        batch_size=1,
        checkpoint_interval=1,
        verbose=False,
    )

    tags.run(args)

    # 验证 .photowall/tags.json 写入成功
    tags_file = tmp_path / ".photowall" / "tags.json"
    assert tags_file.exists()
    final_data = json.loads(tags_file.read_text(encoding="utf-8"))
    assert "pic1.jpg" in final_data["files"]
    assert "pic2.jpg" in final_data["files"]

    # pic1.jpg 保留了断点中的 raw
    assert "cat_ears" in final_data["files"]["pic1.jpg"]["raw"]

    # 并且只对未命中缓存的 pic2.jpg 进行了推理 (mock_session.run 只被调用 1 次)
    assert mock_session.run.call_count == 1

    # 生成完毕后：
    # 1. 超过 7 天的 tags_expired.json 被删除
    assert not expired_cp.exists()
    # 2. 近期有效的 checkpoints 保留（原来的 tags_0001.json + 新增的 pic2.jpg 的 checkpoint）
    remaining_checkpoints = sorted(checkpoint_dir.glob("tags_*.json"))
    assert len(remaining_checkpoints) == 2

    # 再次运行 tags.run：所有图片均命中缓存，不应新增任何 checkpoint
    tags.run(args)
    assert mock_session.run.call_count == 1
    assert len(list(checkpoint_dir.glob("tags_*.json"))) == 2


def test_tags_keyboard_interrupt_saves_remaining_buffer(tmp_path, monkeypatch, capsys):
    assets_dir = tmp_path / "assets"
    img1 = assets_dir / "pic1.jpg"
    img2 = assets_dir / "pic2.jpg"
    _create_dummy_image(img1)
    _create_dummy_image(img2)

    checkpoint_dir = get_checkpoint_dir(tmp_path)

    monkeypatch.setattr(tags, "load_translations", lambda *a: {})
    monkeypatch.setattr(tags, "get_model_and_tags", lambda _: ("mock.onnx", "mock.csv"))
    monkeypatch.setattr(tags, "load_tags_csv", lambda _: (["cat_ears"], [0]))

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 448, 448]
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]

    call_count = 0

    def mock_run_fn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 第一张图片正常推理完成，写入 buffer
            return [np.array([[2.0]])]
        # 第二张图片时模拟用户按下 Ctrl+C
        raise KeyboardInterrupt()

    mock_session.run.side_effect = mock_run_fn
    monkeypatch.setattr(
        tags, "create_inference_session", lambda *a, **kw: (mock_session, "CPU")
    )

    args = argparse.Namespace(
        home=tmp_path,
        dir=tmp_path,
        force=False,
        filter=False,
        device="cpu",
        batch_size=1,
        checkpoint_interval=100,
        verbose=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        tags.run(args)
    assert exc_info.value.code == 130

    # 验证第一张图片已被存入 checkpoint 目录，且仅保存 size/mtime/raw（不含冗余 tags）
    saved_checkpoints = list(checkpoint_dir.glob("tags_*.json"))
    assert len(saved_checkpoints) == 1
    content = json.loads(saved_checkpoints[0].read_text(encoding="utf-8"))
    assert "pic1.jpg" in content
    assert "raw" in content["pic1.jpg"]
    assert "tags" not in content["pic1.jpg"]

    captured = capsys.readouterr()
    assert "已将未满分块的临时进度 (1 张图片) 保存至断点缓存" in captured.out


def test_tags_force_ignores_existing_and_cleans_expired_checkpoints(
    tmp_path, monkeypatch
):
    assets_dir = tmp_path / "assets"
    img1 = assets_dir / "pic1.jpg"
    _create_dummy_image(img1)

    checkpoint_dir = get_checkpoint_dir(tmp_path)
    checkpoint_dir.mkdir(parents=True)
    recent_cp = checkpoint_dir / "tags_recent_old.json"
    recent_cp.write_text(
        json.dumps(
            {
                "pic1.jpg": {
                    "size": img1.stat().st_size,
                    "mtime": int(img1.stat().st_mtime),
                    "raw": {"stale": 0.9},
                }
            }
        ),
        encoding="utf-8",
    )
    expired_cp = checkpoint_dir / "tags_expired_old.json"
    expired_cp.write_text("{}", encoding="utf-8")
    old_ts = datetime.now().timestamp() - 8 * 86400
    os.utime(expired_cp, (old_ts, old_ts))

    monkeypatch.setattr(tags, "load_translations", lambda *a: {})
    monkeypatch.setattr(tags, "get_model_and_tags", lambda _: ("mock.onnx", "mock.csv"))
    monkeypatch.setattr(tags, "load_tags_csv", lambda _: (["cat_ears"], [0]))

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 448, 448]
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.array([[2.0]])]
    monkeypatch.setattr(
        tags, "create_inference_session", lambda *a, **kw: (mock_session, "CPU")
    )

    args = argparse.Namespace(
        home=tmp_path,
        dir=tmp_path,
        force=True,
        filter=False,
        device="cpu",
        batch_size=1,
        checkpoint_interval=100,
        verbose=False,
    )

    tags.run(args)
    # 强制模式重新推理了 pic1.jpg
    assert mock_session.run.call_count == 1
    # 超过 7 天的被清理，近期旧 checkpoint 与新生成的 checkpoint 保留
    assert not expired_cp.exists()
    assert recent_cp.exists()
    assert len(list(checkpoint_dir.glob("tags_*.json"))) == 2


def test_assets_checkpoints_and_incremental_cache(tmp_path):
    """验证 assets 的 checkpoints 生成、合并与增量缓存命中（命中缓存时不新增 checkpoint）。"""
    assets_dir = tmp_path / "assets"
    img1 = assets_dir / "a1.jpg"
    img2 = assets_dir / "a2.jpg"
    img3 = assets_dir / "a3.jpg"
    _create_dummy_image(img1, color=(255, 0, 0))
    _create_dummy_image(img2, color=(0, 255, 0))
    _create_dummy_image(img3, color=(0, 0, 255))

    checkpoint_dir = assets.get_checkpoint_dir(tmp_path)
    assert checkpoint_dir == tmp_path / ".photowall" / "checkpoints" / "assets"

    # 放入一个超过 7 天的过期 checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expired_cp = checkpoint_dir / "assets_expired.json"
    expired_cp.write_text("{}", encoding="utf-8")
    old_ts = datetime.now().timestamp() - 8 * 86400
    os.utime(expired_cp, (old_ts, old_ts))

    # 第一次运行：3 张图，checkpoint_interval=2 -> 生成 2 个分块 (2 + 1)
    args = argparse.Namespace(
        home=tmp_path,
        dir=tmp_path,
        title="测试Assets缓存",
        force=False,
        checkpoint_interval=2,
    )
    assets.run(args)

    assets_json = tmp_path / ".photowall" / "assets.json"
    assert assets_json.exists()
    assert not expired_cp.exists()

    cp_files = sorted(checkpoint_dir.glob("assets_*.json"))
    assert len(cp_files) == 2

    # 第二次运行：全部命中缓存，不新增任何 checkpoint 文件
    assets.run(args)
    assert len(list(checkpoint_dir.glob("assets_*.json"))) == 2

    # 新增第 4 张图：仅新增 1 个包含 a4.jpg 的 checkpoint
    img4 = assets_dir / "a4.jpg"
    _create_dummy_image(img4, color=(128, 128, 128))
    assets.run(args)
    cp_files_after = sorted(checkpoint_dir.glob("assets_*.json"))
    assert len(cp_files_after) == 3
