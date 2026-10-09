from wq_pipeline.client import SimulationSettings
from wq_pipeline.models import GridItem, Template


def test_template_render_substitutes_field():
    t = Template(name="ts_mean_10", expression="ts_mean({f}, 10)", decay=4, truncation=0.01)
    assert t.render("fnd6_pnrsho") == "ts_mean(fnd6_pnrsho, 10)"


def test_grid_item_expression_hash_is_deterministic():
    gi1 = GridItem("t", "fnd6_x", "rank(fnd6_x)", SimulationSettings())
    gi2 = GridItem("t", "fnd6_x", "rank(fnd6_x)", SimulationSettings())
    assert gi1.expression_hash() == gi2.expression_hash()


def test_dedup_key_changes_with_settings():
    base = GridItem("t", "fnd6_x", "rank(fnd6_x)", SimulationSettings(neutralization="MARKET"))
    other = GridItem("t", "fnd6_x", "rank(fnd6_x)", SimulationSettings(neutralization="SUBINDUSTRY"))
    assert base.dedup_key() != other.dedup_key()


def test_dedup_key_stable_across_identical_settings():
    a = GridItem("t", "fnd6_x", "rank(fnd6_x)", SimulationSettings())
    b = GridItem("t", "fnd6_x", "rank(fnd6_x)", SimulationSettings())
    assert a.dedup_key() == b.dedup_key()
