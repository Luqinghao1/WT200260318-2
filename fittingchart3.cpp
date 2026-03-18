/*
 * 文件名: fittingchart3.cpp
 * 文件作用: 标准笛卡尔坐标系图表类实现文件
 * 功能描述:
 * 1. 初始化线性坐标系 (Linear-Linear)。
 * 2. 实现基础图表功能。
 */

#include "fittingchart3.h"
#include "ui_fittingchart3.h"
#include "chartsetting1.h"
#include "modelparameter.h"
#include "styleselectordialog.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>

FittingChart3::FittingChart3(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FittingChart3),
    m_titleElement(nullptr)
{
    ui->setupUi(this);
    m_plot = ui->chart;

    this->setAttribute(Qt::WA_StyledBackground, true);
    this->setStyleSheet("background-color: white;");

    initUi();
    initConnections();
}

FittingChart3::~FittingChart3()
{
    delete ui;
}

void FittingChart3::initUi()
{
    // 配置坐标轴: 双线性
    m_plot->xAxis->setScaleType(QCPAxis::stLinear);
    m_plot->yAxis->setScaleType(QCPAxis::stLinear);

    // 标题布局
    if (m_plot->plotLayout()->rowCount() == 0) m_plot->plotLayout()->insertRow(0);
    m_titleElement = new QCPTextElement(m_plot, "", QFont("Microsoft YaHei", 12, QFont::Bold));
    m_plot->plotLayout()->addElement(0, 0, m_titleElement);

    setupAxisRect();

    m_plot->legend->setVisible(true);
    QFont legendFont("Microsoft YaHei", 9);
    m_plot->legend->setFont(legendFont);
    m_plot->legend->setBrush(QBrush(QColor(255, 255, 255, 200)));
    if (m_plot->axisRect()) {
        m_plot->axisRect()->insetLayout()->addElement(m_plot->legend, Qt::AlignTop | Qt::AlignRight);
    }

    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);
}

void FittingChart3::setupAxisRect()
{
    QCPAxisRect* rect = m_plot->axisRect();
    if (!rect) return;
    QCPAxis *topAxis = rect->axis(QCPAxis::atTop);
    topAxis->setVisible(true);
    topAxis->setTickLabels(false);
    connect(rect->axis(QCPAxis::atBottom), SIGNAL(rangeChanged(QCPRange)), topAxis, SLOT(setRange(QCPRange)));

    QCPAxis *rightAxis = rect->axis(QCPAxis::atRight);
    rightAxis->setVisible(true);
    rightAxis->setTickLabels(false);
    connect(rect->axis(QCPAxis::atLeft), SIGNAL(rangeChanged(QCPRange)), rightAxis, SLOT(setRange(QCPRange)));
}

void FittingChart3::initConnections()
{
    connect(m_plot, &MouseZoom::saveImageRequested, this, &FittingChart3::on_btnSavePic_clicked);
    connect(m_plot, &MouseZoom::exportDataRequested, this, &FittingChart3::on_btnExportData_clicked);
    connect(m_plot, &MouseZoom::settingsRequested, this, &FittingChart3::on_btnSetting_clicked);
    connect(m_plot, &MouseZoom::resetViewRequested, this, &FittingChart3::on_btnReset_clicked);

    connect(m_plot, &MouseZoom::lineStyleRequested, this, &FittingChart3::onLineStyleRequested);
    connect(m_plot, &MouseZoom::deleteSelectedRequested, this, &FittingChart3::onDeleteSelectedRequested);
    connect(m_plot, &MouseZoom::editItemRequested, this, &FittingChart3::onEditItemRequested);
}

void FittingChart3::setTitle(const QString &title)
{
    refreshTitleElement();
    if (m_titleElement) {
        m_titleElement->setText(title);
        m_plot->replot();
    }
}

MouseZoom* FittingChart3::getPlot() { return m_plot; }

void FittingChart3::clearGraphs()
{
    m_plot->clearGraphs();
    m_plot->replot();
}

void FittingChart3::refreshTitleElement()
{
    if (!m_titleElement) {
        if (m_plot->plotLayout()->elementCount() > 0)
            m_titleElement = qobject_cast<QCPTextElement*>(m_plot->plotLayout()->element(0, 0));
    }
}

void FittingChart3::on_btnSavePic_clicked()
{
    QString dir = ModelParameter::instance()->getProjectPath();
    if (dir.isEmpty()) dir = QDir::currentPath();
    QString fileName = QFileDialog::getSaveFileName(this, "保存图片", dir + "/history_chart.png", "PNG (*.png);;JPG (*.jpg);;PDF (*.pdf)");
    if (fileName.isEmpty()) return;
    if (fileName.endsWith(".png")) m_plot->savePng(fileName);
    else if (fileName.endsWith(".jpg")) m_plot->saveJpg(fileName);
    else m_plot->savePdf(fileName);
}

void FittingChart3::on_btnExportData_clicked() { emit exportDataTriggered(); }

void FittingChart3::on_btnSetting_clicked()
{
    refreshTitleElement();
    QString oldTitle = m_titleElement ? m_titleElement->text() : "";
    ChartSetting1 dlg(m_plot, m_titleElement, this);
    dlg.exec();
    refreshTitleElement();
    m_plot->replot();
    if (m_titleElement && m_titleElement->text() != oldTitle) {
        emit titleChanged(m_titleElement->text());
    }
    emit graphsChanged();
}

void FittingChart3::on_btnReset_clicked()
{
    m_plot->rescaleAxes();
    m_plot->replot();
}

void FittingChart3::onLineStyleRequested(QCPItemLine* line) {
    if (!line) return;
    StyleSelectorDialog dlg(StyleSelectorDialog::ModeLine, this);
    dlg.setPen(line->pen());
    if (dlg.exec() == QDialog::Accepted) {
        line->setPen(dlg.getPen());
        m_plot->replot();
    }
}

void FittingChart3::onDeleteSelectedRequested() {
    auto items = m_plot->selectedItems();
    for (auto item : items) m_plot->removeItem(item);
    m_plot->replot();
}

void FittingChart3::onEditItemRequested(QCPAbstractItem* item) {
    if (auto text = qobject_cast<QCPItemText*>(item)) {
        bool ok;
        QString newContent = QInputDialog::getText(this, "修改标注", "内容:", QLineEdit::Normal, text->text(), &ok);
        if (ok && !newContent.isEmpty()) { text->setText(newContent); m_plot->replot(); }
    }
}

void FittingChart3::closeEvent(QCloseEvent *event)
{
    QMessageBox::StandardButton res = QMessageBox::question(this, "确认关闭", "确定要隐藏此图表窗口吗？", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (res == QMessageBox::Yes) event->accept(); else event->ignore();
}
