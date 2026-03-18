/*
 * 文件名: fittingdatadialog.h
 * 文件作用: 拟合数据加载配置窗口头文件
 * 修改记录:
 * 1. [新增] 在 FittingDataSettings 中补充 porosity, thickness 等物理参数字段，
 * 用于在保存/恢复状态时保持完整上下文。
 */

#ifndef FITTINGDATADIALOG_H
#define FITTINGDATADIALOG_H

#include <QDialog>
#include <QMap>
#include <QStandardItemModel>

namespace Ui {
class FittingDataDialog;
}

// 试井类型枚举
enum WellTestType {
    Test_Drawdown, // 压力降落
    Test_Buildup   // 压力恢复
};

// 数据设置结构体
struct FittingDataSettings {
    bool isFromProject;         // 是否来自项目内部数据
    QString projectFileName;    // 项目文件名 (如果是项目数据)
    QString filePath;           // 外部文件路径 (如果是外部数据)

    int timeColIndex;           // 时间列索引
    int pressureColIndex;       // 压力列索引
    int derivColIndex;          // 导数列索引 (-1表示自动计算)
    int skipRows;               // 跳过表头行数

    WellTestType testType;      // 试井类型
    double initialPressure;     // 初始地层压力 (仅降落试井有效)
    double producingTime;       // 关井前生产时间 (仅恢复试井有效)

    // [新增] 补充物理参数，用于状态保存和绘图上下文
    double porosity;            // 孔隙度
    double thickness;           // 厚度 (m)
    double wellRadius;          // 井半径 (m)
    double viscosity;           // 粘度 (mPa.s)
    double ct;                  // 综合压缩系数 (MPa^-1)
    double fvf;                 // 体积系数 (B)
    double rate;                // 产量 (m3/d)

    double lSpacing;            // 导数计算步长
    bool enableSmoothing;       // 是否启用平滑
    int smoothingSpan;          // 平滑窗口

    // 构造函数初始化默认值
    FittingDataSettings() {
        isFromProject = true;
        timeColIndex = 0;
        pressureColIndex = 1;
        derivColIndex = -1;
        skipRows = 1;
        testType = Test_Drawdown;
        initialPressure = 0.0;
        producingTime = 0.0;

        porosity = 0.0;
        thickness = 0.0;
        wellRadius = 0.0;
        viscosity = 0.0;
        ct = 0.0;
        fvf = 0.0;
        rate = 0.0;

        lSpacing = 0.1;
        enableSmoothing = false;
        smoothingSpan = 5;
    }
};

class FittingDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FittingDataDialog(const QMap<QString, QStandardItemModel*>& projectModels, QWidget *parent = nullptr);
    ~FittingDataDialog();

    // 获取设置结果
    FittingDataSettings getSettings() const;

    // 获取预览用的数据模型
    QStandardItemModel* getPreviewModel() const;

private slots:
    void onSourceChanged();
    void onProjectFileSelectionChanged(int index);
    void onBrowseFile();
    void onAccepted();
    void onDerivColumnChanged(int index);
    void onTestTypeChanged();
    void onSmoothingToggled(bool checked);

private:
    Ui::FittingDataDialog *ui;
    QMap<QString, QStandardItemModel*> m_projectDataMap;
    QStandardItemModel* m_fileModel;

    // 解析辅助函数
    bool parseTextFile(const QString& filePath);
    bool parseExcelFile(const QString& filePath);
    QStandardItemModel* getCurrentProjectModel() const;
    void updateColumnComboBoxes(const QStringList& headers);
};

#endif // FITTINGDATADIALOG_H
