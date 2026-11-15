from pathlib import Path

from photowall import index


def test_index_template_card_width_restoration():
    """验证 index.html 内置模板中列宽滑块的持久化与刷新恢复逻辑。"""
    template_path = (
        Path(__file__).resolve().parent.parent / "src" / "photowall" / "index.html"
    )
    assert template_path.exists(), "src/photowall/index.html 文件应当存在"

    html_content = template_path.read_text(encoding="utf-8")

    # 1. HTML 中滑块应包含 card-width-slider 类与 id
    assert 'id="card-width-slider"' in html_content
    assert 'class="card-slider card-width-slider"' in html_content

    # 2. CSS 中应定义 .card-width-slider
    assert ".card-width-slider" in html_content

    # 3. JS 中应包含 localStorage 键 PHOTOWALL_CARD_WIDTH
    assert "PHOTOWALL_CARD_WIDTH" in html_content

    # 4. JS 中应定义并调用 initCardWidth
    assert "function initCardWidth()" in html_content
    assert "initCardWidth();" in html_content

    # 5. 应将 --card-min-width 与 card-width-slider 进行双向同步
    assert "--card-min-width" in html_content
    assert (
        "document.documentElement.style.setProperty('--card-min-width'" in html_content
    )

    # 6. pageshow 事件监听保障 bfcache 及延迟恢复场景
    assert "window.addEventListener('pageshow'" in html_content


def test_copy_index_html_includes_card_width_fix(tmp_path):
    """验证通过 index.copy_index_html 部署的入口文件包含列宽恢复逻辑。"""
    dest = index.copy_index_html(tmp_path)
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert "initCardWidth()" in content
    assert "PHOTOWALL_CARD_WIDTH" in content


def test_index_template_random_sort_option(tmp_path):
    """验证 index.html 内置模板及导出文件包含随机排序选项与打乱逻辑。"""
    template_path = (
        Path(__file__).resolve().parent.parent / "src" / "photowall" / "index.html"
    )
    html_content = template_path.read_text(encoding="utf-8")

    assert '<option value="random">随机排序</option>' in html_content
    assert 'id="random-shuffle-btn"' in html_content
    assert ".random-shuffle-btn.visible" in html_content
    assert "ICONS.shuffle + `<span>重新随机</span>`" in html_content
    assert "function shuffleRandomOrder()" in html_content
    assert "function updateRandomShuffleButton()" in html_content
    assert "function setSortBy(sortBy)" in html_content
    assert "case 'random':" in html_content
    assert "setSortBy(e.target.value);" in html_content

    dest = index.copy_index_html(tmp_path)
    copied_content = dest.read_text(encoding="utf-8")
    assert '<option value="random">随机排序</option>' in copied_content
    assert 'id="random-shuffle-btn"' in copied_content
    assert "shuffleRandomOrder()" in copied_content
