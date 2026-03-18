/*
 * 文件名: wt_datawidget.h
 * 文件作用: 数据编辑器主窗口头文件
 * 功能描述:
 * 1. 定义数据编辑器的主界面类 WT_DataWidget。
 * 2. 负责管理多个 DataSingleSheet 实例，支持多文件同时打开。
 * 3. 协调顶部工具栏与当前活动页签的交互。
 * 4. [更新] 新增 onUnitManager 槽函数声明，处理统一的属性定义和单位换算。
 */

#ifndef WT_DATAWIDGET_H
#define WT_DATAWIDGET_H

#include <QWidget>
#include <QStandardItemModel>
#include <QJsonArray>
#include <QMap>
#include <QVector>
#include <QStringList>
#include "datasinglesheet.h"

namespace Ui {
class WT_DataWidget;
}

class WT_DataWidget : public QWidget
{
    Q_OBJECT

public:
    explicit WT_DataWidget(QWidget *parent = nullptr);
    ~WT_DataWidget();

    void clearAllData();
    void loadFromProjectData();
    QStandardItemModel* getDataModel() const;
    QMap<QString, QStandardItemModel*> getAllDataModels() const;
    void loadData(const QString& filePath, const QString& fileType = "auto");
    QString getCurrentFileName() const;
    bool hasData() const;

signals:
    void dataChanged();
    void fileChanged(const QString& filePath, const QString& fileType);

private slots:
    void onOpenFile();
    void onSave();
    void onExportExcel();
    void onUnitManager(); // [新增] 统一的单位与属性管理按钮槽函数
    void onTimeConvert();
    void onPressureDropCalc();
    void onCalcPwf();
    void onFilterSample();
    void onHighlightErrors();
    void onTabChanged(int index);
    void onTabCloseRequested(int index);
    void onSheetDataChanged();

private:
    Ui::WT_DataWidget *ui;

    void initUI();
    void setupConnections();
    void updateButtonsState();

    void createNewTab(const QString& filePath, const DataImportSettings& settings);
    DataSingleSheet* currentSheet() const;

    // 将完整的全表数据连同表头一并写入文件，防止其余列丢失
    void saveAndLoadNewData(const QString& oldFilePath, const QStringList& headers, const QVector<QStringList>& fullTable);
};

#endif // WT_DATAWIDGET_H
