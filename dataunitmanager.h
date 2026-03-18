/*
 * 文件名: dataunitmanager.h
 * 文件作用: 全局物理量及单位换算管理中心
 * 功能描述:
 * 1. 提供全局单例，集中管理本软件中所有的物理量、单位及换算系数。
 * 2. 采用 $y = kx + b$ (仿射变换) 模型支持带偏移量的精准换算（如温度）。
 * 3. [优化] 细分了同一维度的物理量（如压力、流压、套压分别独立），丰富下拉选项。
 * 4. [优化] 提供获取全库所有单位的接口 (getAllUniqueUnits)，供用户输入未知物理量时兜底选择。
 */

#ifndef DATAUNITMANAGER_H
#define DATAUNITMANAGER_H

#include <QString>
#include <QStringList>
#include <QMap>
#include <QList>
#include <QSet>

enum class UnitSystemType {
    SI_System,      // 理论国际单位制
    Field_System    // 矿场实用制
};

struct UnitDefinition {
    QString unitName;
    double factor;
    double offset;
};

class DataUnitManager
{
public:
    static DataUnitManager* instance();

    // 获取已注册的所有物理量大类名称
    QStringList getRegisteredQuantities() const;

    // 获取全库去重后的所有单位列表 (供未知物理量时选择)
    QStringList getAllUniqueUnits() const;

    // 获取某个物理量对应的所有可用单位列表
    QStringList getUnitsForQuantity(const QString& quantity) const;

    // 获取某个物理量在指定单位制下的默认单位
    QString getDefaultUnit(const QString& quantity, UnitSystemType systemType) const;

    // 核心换算函数：将 fromUnit 下的值转换为 toUnit 下的值
    double convert(double value, const QString& quantity, const QString& fromUnit, const QString& toUnit) const;

private:
    DataUnitManager();
    ~DataUnitManager() = default;

    void registerUnitGroup(const QString& groupName, const QList<UnitDefinition>& units, const QString& siDefault, const QString& fieldDefault);

    QMap<QString, QList<UnitDefinition>> m_unitGroups;   // 物理量名称 -> 包含的单位列表
    QMap<QString, QString> m_siDefaults;                 // 物理量名称 -> SI制默认单位
    QMap<QString, QString> m_fieldDefaults;              // 物理量名称 -> 矿场制默认单位
};

#endif // DATAUNITMANAGER_H
