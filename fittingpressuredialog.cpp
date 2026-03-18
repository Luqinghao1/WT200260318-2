/*
 * 文件名: fittingpressuredialog.cpp
 * 文件作用: 原始地层压力求解配置弹窗实现文件
 */

#include "fittingpressuredialog.h"
#include "ui_fittingpressuredialog.h"

FittingPressureDialog::FittingPressureDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::FittingPressureDialog),
    m_startX(0.0),
    m_endX(0.0)
{
    ui->setupUi(this);
    // 设置窗口属性：置顶显示，非模态
    this->setWindowFlags(Qt::Window | Qt::WindowStaysOnTopHint);
}

FittingPressureDialog::~FittingPressureDialog()
{
    delete ui;
}

void FittingPressureDialog::setStartCoordinate(double x, double y)
{
    m_startX = x;
    // 显示保留4位小数
    ui->le_start->setText(QString("X: %1, Y: %2").arg(QString::number(x, 'f', 4)).arg(QString::number(y, 'f', 4)));
}

void FittingPressureDialog::setEndCoordinate(double x, double y)
{
    m_endX = x;
    ui->le_end->setText(QString("X: %1, Y: %2").arg(QString::number(x, 'f', 4)).arg(QString::number(y, 'f', 4)));
}

double FittingPressureDialog::getStartX() const
{
    return m_startX;
}

double FittingPressureDialog::getEndX() const
{
    return m_endX;
}

void FittingPressureDialog::on_btn_pickStart_clicked()
{
    emit requestPickStart();
}

void FittingPressureDialog::on_btn_pickEnd_clicked()
{
    emit requestPickEnd();
}

void FittingPressureDialog::on_btn_calculate_clicked()
{
    emit requestCalculate();
}

void FittingPressureDialog::on_btn_close_clicked()
{
    this->close();
}
