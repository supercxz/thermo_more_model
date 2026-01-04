from pymatgen.ext.matproj import MPRester

API_KEY = "DEFRMHnLsdudImc5EfW4U18WXglGLKy0"  # 替换为您的实际API密钥

def get_materials_with_thermal_expansion(api_key):
    with MPRester(api_key) as mpr:
        # 检索具有热膨胀系数数据的所有材料
        criteria = {"thermal_expansion_coefficient": {"$exists": True}}
        properties = ["pretty_formula", "thermal_expansion_coefficient"]
        materials_data = mpr.query(criteria=criteria, properties=properties)
        
        # 打印材料的化学式和热膨胀系数
        for material in materials_data:
            formula = material.get("pretty_formula", "N/A")
            tec = material.get("thermal_expansion_coefficient", "N/A")
            print(f"Chemical Formula: {formula}, Thermal Expansion Coefficient: {tec}")

if __name__ == "__main__":
    get_materials_with_thermal_expansion(API_KEY)