/*
 * 文件名: fittingchart2.cpp
 * 文件作用: 半对数坐标系图表类实现文件
 * 修改记录:
 * 1. [修复] onPlotMousePress 中点击空白处时强制恢复 Zoom/Drag 交互。
 * 2. [新增] 拖拽直线时实时计算斜率截距并发射信号。
 */

#include "fittingchart2.h"
#include "ui_fittingchart2.h"
#include "chartsetting1.h"
#include "modelparameter.h"
#include "styleselectordialog.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <cmath>

FittingChart2::FittingChart2(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FittingChart2),
    m_titleElement(nullptr),
    m_interMode(Mode_None),
    m_activeLine(nullptr)
{
    ui->setupUi(this);
    m_plot = ui->chart;

    this->setAttribute(Qt::WA_StyledBackground, true);
    this->setStyleSheet("background-color: white;");

    initUi();
    initConnections();
}

FittingChart2::~FittingChart2()
{
    delete ui;
}

void FittingChart2::initUi()
{
    QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
    m_plot->xAxis->setScaleType(QCPAxis::stLogarithmic);
    m_plot->xAxis->setTicker(logTicker);

    m_plot->yAxis->setScaleType(QCPAxis::stLinear);

    m_plot->xAxis->setNumberFormat("eb");
    m_plot->xAxis->setNumberPrecision(0);

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

void FittingChart2::setupAxisRect()
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

void FittingChart2::initConnections()
{
    connect(m_plot, &MouseZoom::saveImageRequested, this, &FittingChart2::on_btnSavePic_clicked);
    connect(m_plot, &MouseZoom::exportDataRequested, this, &FittingChart2::on_btnExportData_clicked);
    connect(m_plot, &MouseZoom::settingsRequested, this, &FittingChart2::on_btnSetting_clicked);
    connect(m_plot, &MouseZoom::resetViewRequested, this, &FittingChart2::on_btnReset_clicked);

    connect(m_plot, &MouseZoom::lineStyleRequested, this, &FittingChart2::onLineStyleRequested);
    connect(m_plot, &MouseZoom::deleteSelectedRequested, this, &FittingChart2::onDeleteSelectedRequested);
    connect(m_plot, &MouseZoom::editItemRequested, this, &FittingChart2::onEditItemRequested);

    connect(m_plot, &QCustomPlot::mousePress, this, &FittingChart2::onPlotMousePress);
    connect(m_plot, &QCustomPlot::mouseMove, this, &FittingChart2::onPlotMouseMove);
    connect(m_plot, &QCustomPlot::mouseRelease, this, &FittingChart2::onPlotMouseRelease);
    connect(m_plot, &QCustomPlot::mouseDoubleClick, this, &FittingChart2::onPlotMouseDoubleClick);
}

void FittingChart2::setTitle(const QString &title)
{
    refreshTitleElement();
    if (m_titleElement) {
        m_titleElement->setText(title);
        m_plot->replot();
    }
}

MouseZoom* FittingChart2::getPlot() { return m_plot; }

void FittingChart2::clearGraphs()
{
    m_plot->clearGraphs();
    m_plot->replot();
}

void FittingChart2::refreshTitleElement()
{
    if (!m_titleElement) {
        if (m_plot->plotLayout()->elementCount() > 0)
            m_titleElement = qobject_cast<QCPTextElement*>(m_plot->plotLayout()->element(0, 0));
    }
}

void FittingChart2::addStraightLine()
{
    QCPAxisRect* rect = m_plot->axisRect();
    double x1 = rect->axis(QCPAxis::atBottom)->range().lower;
    double x2 = rect->axis(QCPAxis::atBottom)->range().upper;
    double y1 = rect->axis(QCPAxis::atLeft)->range().lower;
    double y2 = rect->axis(QCPAxis::atLeft)->range().upper;

    double midY = (y1 + y2) / 2.0;

    QCPItemLine* line = new QCPItemLine(m_plot);
    line->setClipAxisRect(rect);
    line->start->setCoords(x1 * 1.5, midY);
    line->end->setCoords(x2 / 1.5, midY);

    QPen pen(Qt::blue, 2, Qt::SolidLine);
    line->setPen(pen);
    line->setSelectedPen(QPen(Qt::red, 2, Qt::SolidLine));
    m_plot->replot();

    m_activeLine = line;
    calculateAndEmitLineParams();
    m_activeLine = nullptr;
}

// -----------------------------------------------------------------------------
// 交互核心修复
// -----------------------------------------------------------------------------

void FittingChart2::onPlotMousePress(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton) return;

    m_interMode = Mode_None;
    m_activeLine = nullptr;
    m_lastMousePos = event->pos();
    double tolerance = 8.0;

    // 1. 优先检查线段选中
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        QCPItemLine* line = qobject_cast<QCPItemLine*>(m_plot->item(i));
        if (!line) continue;

        double x1 = m_plot->xAxis->coordToPixel(line->start->coords().x());
        double y1 = m_plot->yAxis->coordToPixel(line->start->coords().y());
        double x2 = m_plot->xAxis->coordToPixel(line->end->coords().x());
        double y2 = m_plot->yAxis->coordToPixel(line->end->coords().y());
        QPointF p(event->pos());

        if (std::sqrt(pow(p.x()-x1,2)+pow(p.y()-y1,2)) < tolerance) {
            m_interMode = Mode_Dragging_Start; m_activeLine = line;
        }
        else if (std::sqrt(pow(p.x()-x2,2)+pow(p.y()-y2,2)) < tolerance) {
            m_interMode = Mode_Dragging_End; m_activeLine = line;
        }
        else if (distToSegment(p, QPointF(x1,y1), QPointF(x2,y2)) < tolerance) {
            m_interMode = Mode_Dragging_Line; m_activeLine = line;
        }

        if (m_interMode != Mode_None) {
            m_plot->deselectAll();
            line->setSelected(true);
            // 选中并拖拽时，禁用 Zoom，防止冲突
            m_plot->setInteractions(QCP::Interaction(0));
            m_plot->replot();
            return;
        }
    }

    // 2. [修复] 点击空白处：取消选择并强制恢复图表交互 (Zoom/Drag)
    m_plot->deselectAll();
    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);
    m_plot->replot();
}

void FittingChart2::onPlotMouseMove(QMouseEvent* event)
{
    if (event->buttons() & Qt::LeftButton) {
        if (m_interMode != Mode_None && m_activeLine) {
            QPointF currentPos = event->pos();
            QPointF delta = currentPos - m_lastMousePos;
            double mouseX = m_plot->xAxis->pixelToCoord(currentPos.x());
            double mouseY = m_plot->yAxis->pixelToCoord(currentPos.y());

            if (m_interMode == Mode_Dragging_Line) {
                double sPx = m_plot->xAxis->coordToPixel(m_activeLine->start->coords().x()) + delta.x();
                double sPy = m_plot->yAxis->coordToPixel(m_activeLine->start->coords().y()) + delta.y();
                double ePx = m_plot->xAxis->coordToPixel(m_activeLine->end->coords().x()) + delta.x();
                double ePy = m_plot->yAxis->coordToPixel(m_activeLine->end->coords().y()) + delta.y();

                m_activeLine->start->setCoords(m_plot->xAxis->pixelToCoord(sPx), m_plot->yAxis->pixelToCoord(sPy));
                m_activeLine->end->setCoords(m_plot->xAxis->pixelToCoord(ePx), m_plot->yAxis->pixelToCoord(ePy));
            }
            else if (m_interMode == Mode_Dragging_Start) {
                m_activeLine->start->setCoords(mouseX, mouseY);
            }
            else if (m_interMode == Mode_Dragging_End) {
                m_activeLine->end->setCoords(mouseX, mouseY);
            }

            // 实时计算并通知参数更新
            calculateAndEmitLineParams();

            m_lastMousePos = currentPos;
            m_plot->replot();
        }
    }
}

void FittingChart2::calculateAndEmitLineParams()
{
    if (!m_activeLine) return;
    double x1 = m_activeLine->start->coords().x();
    double y1 = m_activeLine->start->coords().y();
    double x2 = m_activeLine->end->coords().x();
    double y2 = m_activeLine->end->coords().y();

    if (x1 <= 0) x1 = 1e-5;
    if (x2 <= 0) x2 = 1e-5;

    // k = (y2 - y1) / (log10(x2) - log10(x1))
    double den = log10(x2) - log10(x1);
    if (std::abs(den) < 1e-9) return;

    double k = (y2 - y1) / den;
    // y = k * log10(x) + b => b = y1 - k * log10(x1)
    double b = y1 - k * log10(x1);

    emit sigLineMoved(k, b);
}

void FittingChart2::onPlotMouseRelease(QMouseEvent* event)
{
    Q_UNUSED(event);
    if (m_interMode != Mode_None) {
        // 拖拽结束，恢复交互能力
        m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);
    }
    m_interMode = Mode_None;
}

void FittingChart2::onPlotMouseDoubleClick(QMouseEvent* event) { Q_UNUSED(event); }

double FittingChart2::distToSegment(const QPointF& p, const QPointF& s, const QPointF& e)
{
    double l2 = (s.x()-e.x())*(s.x()-e.x()) + (s.y()-e.y())*(s.y()-e.y());
    if (l2 == 0) return std::sqrt((p.x()-s.x())*(p.x()-s.x()) + (p.y()-s.y())*(p.y()-s.y()));
    double t = ((p.x()-s.x())*(e.x()-s.x()) + (p.y()-s.y())*(e.y()-s.y())) / l2;
    t = std::max(0.0, std::min(1.0, t));
    QPointF proj = s + t * (e - s);
    return std::sqrt((p.x()-proj.x())*(p.x()-proj.x()) + (p.y()-proj.y())*(p.y()-proj.y()));
}

void FittingChart2::on_btnSavePic_clicked()
{
    QString dir = ModelParameter::instance()->getProjectPath();
    if (dir.isEmpty()) dir = QDir::currentPath();
    QString fileName = QFileDialog::getSaveFileName(this, "保存图片", dir + "/semilog_chart.png", "PNG (*.png);;JPG (*.jpg);;PDF (*.pdf)");
    if (fileName.isEmpty()) return;
    if (fileName.endsWith(".png")) m_plot->savePng(fileName);
    else if (fileName.endsWith(".jpg")) m_plot->saveJpg(fileName);
    else m_plot->savePdf(fileName);
}

void FittingChart2::on_btnExportData_clicked() { emit exportDataTriggered(); }

void FittingChart2::on_btnSetting_clicked()
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

void FittingChart2::on_btnReset_clicked()
{
    m_plot->rescaleAxes();
    if(m_plot->xAxis->range().lower <= 0) m_plot->xAxis->setRangeLower(1e-3);
    m_plot->replot();
}

void FittingChart2::onLineStyleRequested(QCPItemLine* line) {
    if (!line) return;
    StyleSelectorDialog dlg(StyleSelectorDialog::ModeLine, this);
    dlg.setPen(line->pen());
    if (dlg.exec() == QDialog::Accepted) {
        line->setPen(dlg.getPen());
        m_plot->replot();
    }
}

void FittingChart2::onDeleteSelectedRequested() {
    auto items = m_plot->selectedItems();
    for (auto item : items) m_plot->removeItem(item);
    m_plot->replot();
}

void FittingChart2::onEditItemRequested(QCPAbstractItem* item) {
    if (auto text = qobject_cast<QCPItemText*>(item)) {
        bool ok;
        QString newContent = QInputDialog::getText(this, "修改标注", "内容:", QLineEdit::Normal, text->text(), &ok);
        if (ok && !newContent.isEmpty()) { text->setText(newContent); m_plot->replot(); }
    }
}

void FittingChart2::closeEvent(QCloseEvent *event)
{
    QMessageBox::StandardButton res = QMessageBox::question(this, "确认关闭", "确定要隐藏此图表窗口吗？", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (res == QMessageBox::Yes) event->accept(); else event->ignore();
}
