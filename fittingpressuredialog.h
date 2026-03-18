/*
 * 文件名: fittingpressuredialog.h
 * 文件作用: 原始地层压力求解配置弹窗头文件
 * 功能描述: 提供用户界面，用于在半对数坐标系上拾取数据点范围，并触发求解计算。
 */

#ifndef FITTINGPRESSUREDIALOG_H
#define FITTINGPRESSUREDIALOG_H

#include <QDialog>

namespace Ui {
class FittingPressureDialog;
}

class FittingPressureDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FittingPressureDialog(QWidget *parent = nullptr);
    ~FittingPressureDialog();

    // 设置起点显示的坐标值
    void setStartCoordinate(double x, double y);
    // 设置终点显示的坐标值
    void setEndCoordinate(double x, double y);

    // 获取用户选择的X轴范围
    double getStartX() const;
    double getEndX() const;

signals:
    // 请求开始拾取起点
    void requestPickStart();
    // 请求开始拾取终点
    void requestPickEnd();
    // 请求执行计算
    void requestCalculate();

private slots:
    void on_btn_pickStart_clicked();
    void on_btn_pickEnd_clicked();
    void on_btn_calculate_clicked();
    void on_btn_close_clicked();

private:
    Ui::FittingPressureDialog *ui;
    double m_startX;
    double m_endX;
};

#endif // FITTINGPRESSUREDIALOG_H
