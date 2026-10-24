import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from photowall import tags
from photowall.cli import build_parser
from photowall.tags import (
    clear_checkpoints,
    get_checkpoint_dir,
    load_checkpoints,
    save_checkpoint,
)


def _create_dummy_image(path: Path, color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path)


def test_save_and_load_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / ".cache" / "tags"

    # 1. 保存分块 1
    chunk_1 = {
        "img1.jpg": {"size": 100, "mtime": 1000, "raw": {"tag1": 0.9}},
        "img2.jpg": {"size": 200, "mtime": 2000, "raw": {"tag2": 0.8}},
    }
    saved_file_1 = save_checkpoint(checkpoint_dir, chunk_1, "chunk_0001")
    assert saved_file_1 is not None
    assert saved_file_1.exists()

    # 2. 保存分块 2
    chunk_2 = {
        "img3.jpg": {"size": 300, "mtime": 3000, "raw": {"tag3": 0.7}},
    }
    saved_file_2 = save_checkpoint(checkpoint_dir, chunk_2, "chunk_0002")
    assert saved_file_2 is not None
    assert saved_file_2.exists()

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


def test_clear_checkpoints(tmp_path):
    checkpoint_dir = tmp_path / ".cache" / "tags"
    checkpoint_dir.mkdir(parents=True)

    (checkpoint_dir / "checkpoint_001.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "checkpoint_002.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "checkpoint_003.tmp").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "keep_me.txt").write_text("keep", encoding="utf-8")

    removed = clear_checkpoints(checkpoint_dir)
    assert removed == 3
    assert not (checkpoint_dir / "checkpoint_001.json").exists()
    assert not (checkpoint_dir / "checkpoint_002.json").exists()
    assert not (checkpoint_dir / "checkpoint_003.tmp").exists()
    assert (checkpoint_dir / "keep_me.txt").exists()


def test_load_checkpoints_with_corrupted_file(tmp_path, capsys):
    checkpoint_dir = tmp_path / ".cache" / "tags"
    checkpoint_dir.mkdir(parents=True)

    # 正常文件
    valid_data = {"img1.jpg": {"size": 100, "mtime": 1000, "raw": {}}}
    (checkpoint_dir / "checkpoint_001.json").write_text(
        json.dumps(valid_data), encoding="utf-8"
    )

    # 损坏文件
    (checkpoint_dir / "checkpoint_corrupted.json").write_text(
        "{ bad_json ...", encoding="utf-8"
    )

    loaded = load_checkpoints(checkpoint_dir)
    assert len(loaded) == 1
    assert "img1.jpg" in loaded

    captured = capsys.readouterr()
    assert "读取断点缓存文件 checkpoint_corrupted.json 失败" in captured.out


def test_cli_checkpoint_interval_args():
    parser = build_parser()
    args_tags = parser.parse_args(["tags", "--checkpoint-interval", "50"])
    assert args_tags.checkpoint_interval == 50

    args_index = parser.parse_args(["index", "--checkpoint-interval", "200"])
    assert args_index.checkpoint_interval == 200


def test_tags_resume_from_checkpoint_and_cleanup(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    img1 = assets_dir / "pic1.jpg"
    img2 = assets_dir / "pic2.jpg"
    _create_dummy_image(img1)
    _create_dummy_image(img2)

    stat1 = img1.stat()

    # 模拟在中断前，pic1.jpg 已经存在断点缓存中
    checkpoint_dir = tmp_path / ".cache" / "tags"
    checkpoint_dir.mkdir(parents=True)
    cp_data = {
        "pic1.jpg": {
            "size": stat1.st_size,
            "mtime": int(stat1.st_mtime),
            "raw": {"cat_ears": 0.95},
        }
    }
    (checkpoint_dir / "checkpoint_0001.json").write_text(
        json.dumps(cp_data), encoding="utf-8"
    )

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

    # 验证 tags.json 写入成功
    tags_file = tmp_path / "tags.json"
    assert tags_file.exists()
    final_data = json.loads(tags_file.read_text(encoding="utf-8"))
    assert "pic1.jpg" in final_data["files"]
    assert "pic2.jpg" in final_data["files"]

    # pic1.jpg 保留了断点中的 raw
    assert "cat_ears" in final_data["files"]["pic1.jpg"]["raw"]

    # 并且只对 pic2.jpg 进行了推理 (mock_session.run 只被调用 1 次)
    assert mock_session.run.call_count == 1

    # 生成完毕后，断点缓存目录应被清空
    remaining_checkpoints = list(checkpoint_dir.glob("*.json"))
    assert len(remaining_checkpoints) == 0


def test_tags_keyboard_interrupt_saves_remaining_buffer(tmp_path, monkeypatch):
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

    # 验证第一张图片已被存入 checkpoint 目录
    saved_checkpoints = list(checkpoint_dir.glob("*.json"))
    assert len(saved_checkpoints) == 1
    content = json.loads(saved_checkpoints[0].read_text(encoding="utf-8"))
    assert "pic1.jpg" in content


def test_tags_force_clears_existing_checkpoints(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    img1 = assets_dir / "pic1.jpg"
    _create_dummy_image(img1)

    checkpoint_dir = tmp_path / ".cache" / "tags"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "old_checkpoint.json").write_text("{}", encoding="utf-8")

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
    import numpy as np

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
    assert not (checkpoint_dir / "old_checkpoint.json").exists()
