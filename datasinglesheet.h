/*
 * 文件名: datasinglesheet.h
 * 文件作用: 单个数据表页签类头文件
 * 功能描述:
 * 1. 管理单个数据文件的显示(QTableView)和数据模型(QStandardItemModel)。
 * 2. 处理该页签内的数据加载、计算、右键菜单操作。
 * 3. 支持 Ctrl+滚轮 缩放表格。
 * 4. 提供数据的序列化(JSON)和反序列化接口。
 * 5. [更新] 移除了废弃的 onDefineColumns 声明，适配全新的全局单位与属性管理架构。
 */

#ifndef DATASINGLESHEET_H
#define DATASINGLESHEET_H

#include <QWidget>
#include <QStandardItemModel>
#include <QSortFilterProxyModel>
#include <QUndoStack>
#include <QStyledItemDelegate>
#include <QMenu>
#include <QJsonArray>
#include <QJsonObject>
#include "dataimportdialog.h"

// 试井物理量枚举定义
enum class WellTestColumnType {
    SerialNumber, Date, Time, TimeOfDay, Pressure, CasingPressure, BottomHolePressure,
    Temperature, FlowRate, Depth, Viscosity, Density, Permeability, Porosity, WellRadius,
    SkinFactor, Distance, Volume, PressureDrop, Custom
};

// 列属性定义结构体
struct ColumnDefinition {
    QString name;
    WellTestColumnType type;
    QString unit;
    bool isRequired;
    int decimalPlaces;

    ColumnDefinition() : type(WellTestColumnType::Custom), isRequired(false), decimalPlaces(3) {}
};

namespace Ui {
class DataSingleSheet;
}

// 拦截右键菜单的代理类
class NoContextMenuDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit NoContextMenuDelegate(QObject *parent = nullptr) : QStyledItemDelegate(parent) {}
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const override;
};

class DataSingleSheet : public QWidget
{
    Q_OBJECT

public:
    explicit DataSingleSheet(QWidget *parent = nullptr);
    ~DataSingleSheet();

    bool loadData(const QString& filePath, const DataImportSettings& settings);
    void loadFromJson(const QJsonObject& jsonSheet);
    QJsonObject saveToJson() const;

    QString getFilePath() const { return m_filePath; }
    void setFilePath(const QString& path) { m_filePath = path; }
    QStandardItemModel* getDataModel() const { return m_dataModel; }
    void setFilterText(const QString& text);

protected:
    // 事件过滤器，用于处理 Ctrl+滚轮 缩放
    bool eventFilter(QObject *obj, QEvent *event) override;

public slots:
    void onExportExcel();
    void onTimeConvert();
    void onPressureDropCalc();
    void onCalcPwf();
    void onHighlightErrors();

    void onCustomContextMenu(const QPoint& pos);
    void onMergeCells();
    void onUnmergeCells();
    void onSortAscending();
    void onSortDescending();
    void onSplitColumn();

    void onAddRow(int insertMode = 0);
    void onDeleteRow();
    void onHideRow();
    void onShowAllRows();
    void onAddCol(int insertMode = 0);
    void onDeleteCol();
    void onHideCol();
    void onShowAllCols();

signals:
    void dataChanged();

private slots:
    void onModelDataChanged();

private:
    Ui::DataSingleSheet *ui;

    QStandardItemModel* m_dataModel;
    QSortFilterProxyModel* m_proxyModel;
    QUndoStack* m_undoStack;

    QString m_filePath;
    QList<ColumnDefinition> m_columnDefinitions;

    void initUI();
    void setupModel();

    bool loadExcelFile(const QString& path, const DataImportSettings& settings);
    bool loadTextFile(const QString& path, const DataImportSettings& settings);

    QJsonArray serializeRows() const;
    void deserializeRows(const QJsonArray& array);
};

#endif // DATASINGLESHEET_H
