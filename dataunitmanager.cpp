/*
 * 文件名: dataunitmanager.cpp
 * 文件作用: 全局物理量及单位换算管理中心实现文件
 * 功能描述:
 * 1. 初始化预置的物理量和单位定义。
 * 2. [优化] 取消了别名映射，将原有的（压力/流压/套压）拆分为独立的三个物理量名称注册，使其都能在下拉框中显示。
 * 3. 实现了全库单位的汇总提取功能。
 */

#include "dataunitmanager.h"
#include <cmath>
#include <algorithm>

DataUnitManager* DataUnitManager::instance()
{
    static DataUnitManager mgr;
    return &mgr;
}

DataUnitManager::DataUnitManager()
{
    // 1. 压力组 (拆分为多个独立物理量名称，但共享单位与换算)
    QList<UnitDefinition> pUnits = {
        {"Pa", 1.0, 0.0}, {"kPa", 1e3, 0.0}, {"MPa", 1e6, 0.0},
        {"psi", 6894.757, 0.0}, {"bar", 1e5, 0.0}, {"atm", 101325.0, 0.0}
    };
    registerUnitGroup("压力", pUnits, "Pa", "MPa");
    registerUnitGroup("流压", pUnits, "Pa", "MPa");
    registerUnitGroup("套压", pUnits, "Pa", "MPa");
    registerUnitGroup("压降", pUnits, "Pa", "MPa");

    // 2. 温度组
    QList<UnitDefinition> tUnits = {
        {"K", 1.0, 0.0}, {"°C", 1.0, 273.15}, {"°F", 5.0 / 9.0, 255.372222}
    };
    registerUnitGroup("温度", tUnits, "K", "°C");

    // 3. 时间组
    QList<UnitDefinition> timeUnits = {
        {"s", 1.0, 0.0}, {"min", 60.0, 0.0}, {"h", 3600.0, 0.0}, {"d", 86400.0, 0.0}
    };
    registerUnitGroup("时间", timeUnits, "s", "h");
    registerUnitGroup("时刻", timeUnits, "s", "h");

    // 4. 长度组 (细分)
    QList<UnitDefinition> lenUnits = {
        {"m", 1.0, 0.0}, {"cm", 0.01, 0.0}, {"mm", 0.001, 0.0},
        {"km", 1000.0, 0.0}, {"ft", 0.3048, 0.0}, {"in", 0.0254, 0.0}
    };
    registerUnitGroup("长度", lenUnits, "m", "m");
    registerUnitGroup("深度", lenUnits, "m", "m");
    registerUnitGroup("距离", lenUnits, "m", "m");
    registerUnitGroup("井半径", lenUnits, "m", "m");

    // 5. 流量组
    QList<UnitDefinition> flowUnits = {
        {"m³/s", 1.0, 0.0}, {"m³/d", 1.0 / 86400.0, 0.0},
        {"m³/h", 1.0 / 3600.0, 0.0}, {"L/s", 0.001, 0.0}, {"bbl/d", 0.158987/86400.0, 0.0}
    };
    registerUnitGroup("流量", flowUnits, "m³/s", "m³/d");
    registerUnitGroup("产气量", flowUnits, "m³/s", "m³/d");

    // 6. 渗透率组
    QList<UnitDefinition> permUnits = {
        {"m²", 1.0, 0.0}, {"μm²", 1e-12, 0.0}, {"mD", 0.9869233e-15, 0.0}, {"D", 0.9869233e-12, 0.0}
    };
    registerUnitGroup("渗透率", permUnits, "m²", "mD");

    // 7. 粘度组
    QList<UnitDefinition> viscUnits = {
        {"Pa·s", 1.0, 0.0}, {"mPa·s", 0.001, 0.0}, {"cP", 0.001, 0.0}
    };
    registerUnitGroup("粘度", viscUnits, "Pa·s", "mPa·s");

    // 8. 密度组
    QList<UnitDefinition> denUnits = {
        {"kg/m³", 1.0, 0.0}, {"g/cm³", 1000.0, 0.0}, {"lb/ft³", 16.0185, 0.0}
    };
    registerUnitGroup("密度", denUnits, "kg/m³", "g/cm³");

    // 9. 无量纲比例组
    QList<UnitDefinition> fracUnits = {
        {"fraction", 1.0, 0.0}, {"%", 0.01, 0.0}
    };
    registerUnitGroup("孔隙度", fracUnits, "fraction", "fraction");
    registerUnitGroup("表皮系数", fracUnits, "fraction", "fraction");

    // 10. 体积组
    QList<UnitDefinition> volUnits = {
        {"m³", 1.0, 0.0}, {"L", 0.001, 0.0}, {"bbl", 0.158987, 0.0}, {"ft³", 0.0283168, 0.0}
    };
    registerUnitGroup("体积", volUnits, "m³", "m³");
    registerUnitGroup("累产", volUnits, "m³", "m³");
}

void DataUnitManager::registerUnitGroup(const QString& groupName, const QList<UnitDefinition>& units, const QString& siDefault, const QString& fieldDefault)
{
    m_unitGroups[groupName] = units;
    m_siDefaults[groupName] = siDefault;
    m_fieldDefaults[groupName] = fieldDefault;
}

QStringList DataUnitManager::getRegisteredQuantities() const
{
    return m_unitGroups.keys();
}

QStringList DataUnitManager::getAllUniqueUnits() const
{
    QSet<QString> uniqueUnits;
    for (const auto& list : m_unitGroups.values()) {
        for (const auto& def : list) {
            uniqueUnits.insert(def.unitName);
        }
    }
    QStringList result(uniqueUnits.begin(), uniqueUnits.end());
    result.sort();
    return result;
}

QStringList DataUnitManager::getUnitsForQuantity(const QString& quantity) const
{
    QStringList units;
    if (m_unitGroups.contains(quantity)) {
        for (const auto& def : m_unitGroups[quantity]) {
            units.append(def.unitName);
        }
    }
    return units;
}

QString DataUnitManager::getDefaultUnit(const QString& quantity, UnitSystemType systemType) const
{
    if (systemType == UnitSystemType::SI_System && m_siDefaults.contains(quantity)) {
        return m_siDefaults[quantity];
    } else if (systemType == UnitSystemType::Field_System && m_fieldDefaults.contains(quantity)) {
        return m_fieldDefaults[quantity];
    }
    return "";
}

double DataUnitManager::convert(double value, const QString& quantity, const QString& fromUnit, const QString& toUnit) const
{
    if (fromUnit == toUnit || fromUnit.isEmpty() || toUnit.isEmpty()) return value;

    // 寻找物理量及其单位定义
    if (!m_unitGroups.contains(quantity)) return value;
    const QList<UnitDefinition>& units = m_unitGroups[quantity];

    UnitDefinition fromDef{"", 1.0, 0.0}, toDef{"", 1.0, 0.0};
    bool foundFrom = false, foundTo = false;

    for (const auto& def : units) {
        if (def.unitName == fromUnit) { fromDef = def; foundFrom = true; }
        if (def.unitName == toUnit) { toDef = def; foundTo = true; }
    }

    if (!foundFrom || !foundTo) return value;

    double baseValue = value * fromDef.factor + fromDef.offset;
    double targetValue = (baseValue - toDef.offset) / toDef.factor;

    return targetValue;
}
