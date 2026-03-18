/*
 * 文件名: datacalculate.h
 * 文件作用: 数据计算处理类头文件
 * 功能描述:
 * 1. 包含时间转换的配置对话框类 TimeConversionDialog。
 * 2. 包含井底流压计算配置对话框类 PwfCalculationDialog (新增)。
 * 3. 提供 DataCalculate 类，用于执行时间格式转换、压降计算和井底流压计算逻辑。
 * 4. 所有的计算操作都直接修改传入的 QStandardItemModel。
 */

#ifndef DATACALCULATE_H
#define DATACALCULATE_H

#include <QObject>
#include <QDialog>
#include <QStandardItemModel>
#include <QRadioButton>
#include <QComboBox>
#include <QLineEdit>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include "wt_datawidget.h" // 获取相关结构体定义

// 时间转换配置结构体
struct TimeConversionConfig {
    int dateColumnIndex;
    int timeColumnIndex;
    int sourceTimeColumnIndex;
    QString outputUnit;
    QString newColumnName;
    bool useDateAndTime;
};

// 压降计算结果结构体
struct PressureDropResult {
    bool success;
    QString errorMessage;
    int addedColumnIndex;
    QString columnName;
    int processedRows;
};

// 时间转换结果结构体
struct TimeConversionResult {
    bool success;
    QString errorMessage;
    int addedColumnIndex;
    QString columnName;
    int processedRows;
};

// 井底流压计算参数配置结构体
struct PwfCalculationConfig {
    double Hres;         // 油层中部深度 (m)
    double gamma_o;      // 油比重 (g/cm³)
    double gamma_w;      // 水比重 (g/cm³)
    double f_w;          // 质量含水率 (%)
    int pcColumnIndex;   // 套压列索引
    int lwfColumnIndex;  // 动液面列索引
    int decimalPlaces;   // 保留小数位数 (新增)
};

// 井底流压计算结果结构体
struct PwfCalculationResult {
    bool success;
    QString errorMessage;
    int addedColumnIndex;
};

// ============================================================================
// 时间转换设置对话框类
// ============================================================================
class TimeConversionDialog : public QDialog
{
    Q_OBJECT
public:
    explicit TimeConversionDialog(const QStringList& columnNames, QWidget* parent = nullptr);
    TimeConversionConfig getConversionConfig() const;

private slots:
    void onPreviewClicked();
    void onConversionModeChanged();

private:
    void setupUI();
    void updateUIForMode();
    // 辅助函数：生成预览字符串
    QString previewConversion(const QString& d, const QString& t, const QString& unit);

    QStringList m_columnNames;
    QRadioButton* m_dateTimeRadio;
    QRadioButton* m_timeOnlyRadio;
    QComboBox* m_dateColumnCombo;
    QComboBox* m_timeColumnCombo;
    QComboBox* m_sourceColumnCombo;
    QComboBox* m_outputUnitCombo;
    QLineEdit* m_newColumnNameEdit;
    QLabel* m_previewLabel;
};

// ============================================================================
// 井底流压计算参数设置对话框类
// ============================================================================
class PwfCalculationDialog : public QDialog
{
    Q_OBJECT
public:
    explicit PwfCalculationDialog(const QStringList& columnNames, QWidget* parent = nullptr);
    PwfCalculationConfig getConfig() const;

private:
    QStringList m_columnNames;
    QDoubleSpinBox* m_spinHres;    // 油层中部深度
    QDoubleSpinBox* m_spinGammaO;  // 油比重
    QDoubleSpinBox* m_spinGammaW;  // 水比重
    QDoubleSpinBox* m_spinFw;      // 含水率
    QComboBox* m_comboPc;          // 套压列选择
    QComboBox* m_comboLwf;         // 动液面列选择
    QSpinBox* m_spinDecimal;       // 小数位数选择 (新增)
};

// ============================================================================
// 数据计算逻辑处理类
// ============================================================================
class DataCalculate : public QObject
{
    Q_OBJECT
public:
    explicit DataCalculate(QObject* parent = nullptr);

    // 执行时间转换逻辑
    TimeConversionResult convertTimeColumn(QStandardItemModel* model,
                                           QList<ColumnDefinition>& definitions,
                                           const TimeConversionConfig& config);

    // 执行压降计算逻辑
    PressureDropResult calculatePressureDrop(QStandardItemModel* model,
                                             QList<ColumnDefinition>& definitions);

    // 执行井底流压计算逻辑
    PwfCalculationResult calculateBottomHolePressure(QStandardItemModel* model,
                                                     QList<ColumnDefinition>& definitions,
                                                     const PwfCalculationConfig& config);

private:
    // 辅助函数：时间解析
    QTime parseTimeString(const QString& timeStr) const;
    QDate parseDateString(const QString& dateStr) const;
    QDateTime combineDateAndTime(const QDate& date, const QTime& time) const;
    double convertTimeToUnit(double seconds, const QString& unit) const;

    // 辅助函数：查找压力列
    int findPressureColumn(QStandardItemModel* model, const QList<ColumnDefinition>& definitions) const;
};

#endif // DATACALCULATE_H
