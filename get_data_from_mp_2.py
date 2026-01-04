# pip install mp_api

# 简单样例
# from mp_api.client import MPRester

# with MPRester("DEFRMHnLsdudImc5EfW4U18WXglGLKy0") as mpr:
#     docs = mpr.materials.summary.search(material_ids=["mp-149"], fields=["structure"])
#     structure = docs[0].structure

from mp_api.client import MPRester
import pandas as pd

# 初始化 MPRester（请替换为你的 API 密钥）
with MPRester("DEFRMHnLsdudImc5EfW4U18WXglGLKy0") as mpr:
    # 查询含 Li 的无机化合物，提取与带隙相关的性质
    docs = mpr.materials.summary.search(
        elements=["Li"],  # 包含 Li 的材料
        fields=[
            "material_id",  # 唯一标识符
            "band_gap",  # 目标变量：带隙
            "structure",  # 晶体结构（用于提取晶格参数）
            "spacegroup",  # 空间群信息
            "density",  # 密度
            "volume",  # 单胞体积
            "formation_energy_per_atom",  # 形成能
            "e_above_hull",  # 稳定性指标
            "total_magnetization",  # 总磁化强度
            "elements",  # 元素组成
            "nelements",  # 元素种类数
            "nsites",  # 原子总数
            "composition",  # 化学式
        ]
    )

# 提取数据并构建 DataFrame
data = []
for doc in docs:
    if doc.band_gap is None:
        continue  # 跳过带隙缺失的条目

    structure = doc.structure
    data.append({
        "material_id": doc.material_id,
        "band_gap": doc.band_gap,
        "spacegroup": doc.spacegroup.symbol,
        "density": doc.density,
        "volume": doc.volume,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "e_above_hull": doc.e_above_hull,
        "total_magnetization": doc.total_magnetization,
        "nelements": doc.nelements,
        "nsites": doc.nsites,
        "chemical_formula": str(doc.composition.reduced_formula),
        "lattice_a": structure.lattice.a,
        "lattice_b": structure.lattice.b,
        "lattice_c": structure.lattice.c,
        "lattice_alpha": structure.lattice.alpha,
        "lattice_beta": structure.lattice.beta,
        "lattice_gamma": structure.lattice.gamma,
    })

print(f"共提取 {len(data)} 条含 Li 化合物的带隙相关性质。")

# 保存为 CSV 文件（用于后续建模）
# df = pd.DataFrame(data)
# df.to_csv("li_compounds_bandgap_features.csv", index=False)

# print(f"共提取 {len(df)} 条含 Li 化合物的带隙相关性质。")
