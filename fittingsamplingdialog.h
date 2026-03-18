/*
 * 文件名: fittingsamplingdialog.h
 * 文件作用: 定义数据抽样设置对话框及相关数据结构
 * 功能描述:
 * 1. 定义 SamplingInterval 结构体，用于存储抽样区间信息。
 * 2. 定义 SamplingSettingsDialog 类，提供用户交互界面以设置自定义抽样策略。
 */

#ifndef FITTINGSAMPLINGDIALOG_H
#define FITTINGSAMPLINGDIALOG_H

#include <QDialog>
#include <QTableWidget>
#include <QCheckBox>
#include <QList>

// 抽样区间结构体
struct SamplingInterval {
    double tStart; // 起始时间
    double tEnd;   // 结束时间
    int count;     // 该区间内的抽样点数
};

class SamplingSettingsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit SamplingSettingsDialog(const QList<SamplingInterval>& intervals, bool enabled,
                                    double dataMinT, double dataMaxT, QWidget *parent = nullptr);

    QList<SamplingInterval> getIntervals() const;
    bool isCustomSamplingEnabled() const;

private slots:
    void onAddRow();      // 添加一行
    void onRemoveRow();   // 删除选中行
    void onResetDefault();// 重置为默认区间 (按对数空间)

private:
    QTableWidget* m_table; // 表格控件
    QCheckBox* m_chkEnable;// 启用开关
    double m_dataMinT;     // 数据最小时间
    double m_dataMaxT;     // 数据最大时间

    void addRow(double start, double end, int count);
};

#endif // FITTINGSAMPLINGDIALOG_H
