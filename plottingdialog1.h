/*
 * 文件名: plottingdialog1.h
 * 文件作用: 新建单一曲线配置对话框头文件
 * 功能描述:
 * 1. 设置曲线基础信息（名称、数据源）。
 * 2. 选择X/Y数据列。
 * 3. 样式设置：纯中文界面，支持图标预览。
 * 4. 默认名称逻辑：默认为纵轴数据列名（取 \ 前内容）。
 * 5. 数据来源后缀格式改为 (文件名)。
 */

#ifndef PLOTTINGDIALOG1_H
#define PLOTTINGDIALOG1_H

#include <QDialog>
#include <QStandardItemModel>
#include <QColor>
#include <QMap>
#include <QComboBox>
#include "qcustomplot.h"

namespace Ui {
class PlottingDialog1;
}

class PlottingDialog1 : public QDialog
{
    Q_OBJECT

public:
    // 构造函数接收所有数据模型的映射表
    explicit PlottingDialog1(const QMap<QString, QStandardItemModel*>& models, QWidget *parent = nullptr);
    ~PlottingDialog1();

    // --- 获取用户配置 ---
    QString getCurveName() const;
    QString getLegendName() const;      // 图例名称默认为 Y 轴列名
    QString getSelectedFileName() const;// 获取用户选择的数据源文件名

    // 获取数据列索引
    int getXColumn() const;
    int getYColumn() const;

    // 获取标签（默认为列名）
    QString getXLabel() const;
    QString getYLabel() const;

    // --- 样式获取 ---
    QCPScatterStyle::ScatterShape getPointShape() const;
    QColor getPointColor() const; // 获取点颜色下拉框的值

    Qt::PenStyle getLineStyle() const;
    QColor getLineColor() const;  // 获取线颜色下拉框的值
    int getLineWidth() const;

    // 是否在新窗口显示
    bool isNewWindow() const;

private slots:
    // 数据源文件改变时触发
    void onFileChanged(int index);

    // [新增] Y轴列改变时触发（用于更新默认名称）
    void onYColumnChanged(int index);

    // 显示数据来源复选框状态改变时触发
    void onShowSourceChanged(bool checked);

private:
    Ui::PlottingDialog1 *ui;

    // 存储所有可用模型
    QMap<QString, QStandardItemModel*> m_dataMap;
    // 当前选中的模型指针
    QStandardItemModel* m_currentModel;

    // 移除了静态计数器，因为名称由列名决定

    // 记录上一次自动追加的后缀，用于更新或移除
    QString m_lastSuffix;

    // 辅助函数
    void populateComboBoxes(); // 根据当前模型初始化数据列下拉框
    void setupStyleUI();       // 初始化样式控件（填充下拉框内容）
    void initColorComboBox(QComboBox* combo); // 通用函数：填充颜色下拉框

    // 生成基础名称（基于Y轴列名）
    void updateBaseName();
    // 更新曲线名称后缀
    void updateNameSuffix();

    // 图标生成辅助函数
    QIcon createPointIcon(QCPScatterStyle::ScatterShape shape);
    QIcon createLineIcon(Qt::PenStyle style);
};

#endif // PLOTTINGDIALOG1_H
