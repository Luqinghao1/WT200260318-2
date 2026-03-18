/*
 * 文件名: fittingchart1.cpp
 * 文件作用: 双对数坐标系图表类实现文件
 * 功能描述:
 * 1. 初始化双对数坐标系 (Log-Log Axis)。
 * 2. 实现特征线 (Unit Slope, Half Slope 等) 的绘制逻辑。
 * 3. 实现图表交互逻辑。
 */

#include "fittingchart1.h"
#include "ui_fittingchart1.h"
#include "chartsetting1.h"
#include "modelparameter.h"
#include "styleselectordialog.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <cmath>

FittingChart1::FittingChart1(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FittingChart1),
    m_titleElement(nullptr),
    m_interMode(Mode_None),
    m_activeLine(nullptr),
    m_activeText(nullptr),
    m_activeArrow(nullptr)
{
    ui->setupUi(this);
    m_plot = ui->chart; // 假设UI文件中控件名为 chart，且提升为 MouseZoom

    // 强制背景为白色
    this->setAttribute(Qt::WA_StyledBackground, true);
    this->setStyleSheet("background-color: white;");

    initUi();
    initConnections();
}

FittingChart1::~FittingChart1()
{
    delete ui;
}

void FittingChart1::initUi()
{
    // 配置坐标轴为双对数
    QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
    m_plot->xAxis->setScaleType(QCPAxis::stLogarithmic);
    m_plot->xAxis->setTicker(logTicker);
    m_plot->yAxis->setScaleType(QCPAxis::stLogarithmic);
    m_plot->yAxis->setTicker(logTicker);

    m_plot->xAxis->setNumberFormat("eb");
    m_plot->xAxis->setNumberPrecision(0);
    m_plot->yAxis->setNumberFormat("eb");
    m_plot->yAxis->setNumberPrecision(0);

    // 标题布局
    if (m_plot->plotLayout()->rowCount() == 0) m_plot->plotLayout()->insertRow(0);
    m_titleElement = new QCPTextElement(m_plot, "", QFont("Microsoft YaHei", 12, QFont::Bold));
    m_plot->plotLayout()->addElement(0, 0, m_titleElement);

    // 坐标轴矩形设置
    setupAxisRect();

    // 图例设置
    m_plot->legend->setVisible(true);
    QFont legendFont("Microsoft YaHei", 9);
    m_plot->legend->setFont(legendFont);
    m_plot->legend->setBrush(QBrush(QColor(255, 255, 255, 200)));
    if (m_plot->axisRect()) {
        m_plot->axisRect()->insetLayout()->addElement(m_plot->legend, Qt::AlignTop | Qt::AlignRight);
    }

    // 初始化右键菜单 (特征线)
    m_lineMenu = new QMenu(this);
    QAction* actSlope1 = m_lineMenu->addAction("斜率 k = 1 (井筒储集)");
    connect(actSlope1, &QAction::triggered, this, [=](){ addCharacteristicLine(1.0); });

    QAction* actSlopeHalf = m_lineMenu->addAction("斜率 k = 1/2 (线性流)");
    connect(actSlopeHalf, &QAction::triggered, this, [=](){ addCharacteristicLine(0.5); });

    QAction* actSlopeQuarter = m_lineMenu->addAction("斜率 k = 1/4 (双线性流)");
    connect(actSlopeQuarter, &QAction::triggered, this, [=](){ addCharacteristicLine(0.25); });

    QAction* actHorizontal = m_lineMenu->addAction("水平线 (径向流)");
    connect(actHorizontal, &QAction::triggered, this, [=](){ addCharacteristicLine(0.0); });

    // 默认交互模式
    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);
}

void FittingChart1::setupAxisRect()
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

void FittingChart1::initConnections()
{
    // 连接 MouseZoom 信号
    connect(m_plot, &MouseZoom::saveImageRequested, this, &FittingChart1::on_btnSavePic_clicked);
    connect(m_plot, &MouseZoom::exportDataRequested, this, &FittingChart1::on_btnExportData_clicked);
    // 双对数图特有：右键绘制斜率线
    connect(m_plot, &MouseZoom::drawLineRequested, this, &FittingChart1::addCharacteristicLine);
    connect(m_plot, &MouseZoom::settingsRequested, this, &FittingChart1::on_btnSetting_clicked);
    connect(m_plot, &MouseZoom::resetViewRequested, this, &FittingChart1::on_btnReset_clicked);

    connect(m_plot, &MouseZoom::addAnnotationRequested, this, &FittingChart1::onAddAnnotationRequested);
    connect(m_plot, &MouseZoom::lineStyleRequested, this, &FittingChart1::onLineStyleRequested);
    connect(m_plot, &MouseZoom::deleteSelectedRequested, this, &FittingChart1::onDeleteSelectedRequested);
    connect(m_plot, &MouseZoom::editItemRequested, this, &FittingChart1::onEditItemRequested);

    // 原生鼠标事件
    connect(m_plot, &QCustomPlot::mousePress, this, &FittingChart1::onPlotMousePress);
    connect(m_plot, &QCustomPlot::mouseMove, this, &FittingChart1::onPlotMouseMove);
    connect(m_plot, &QCustomPlot::mouseRelease, this, &FittingChart1::onPlotMouseRelease);
    connect(m_plot, &QCustomPlot::mouseDoubleClick, this, &FittingChart1::onPlotMouseDoubleClick);
}

void FittingChart1::setTitle(const QString &title)
{
    refreshTitleElement();
    if (m_titleElement) {
        m_titleElement->setText(title);
        m_plot->replot();
    }
}

MouseZoom* FittingChart1::getPlot() { return m_plot; }

void FittingChart1::clearGraphs()
{
    m_plot->clearGraphs();
    m_plot->replot();
}

void FittingChart1::refreshTitleElement()
{
    if (!m_titleElement) {
        if (m_plot->plotLayout()->elementCount() > 0)
            m_titleElement = qobject_cast<QCPTextElement*>(m_plot->plotLayout()->element(0, 0));
    }
}

// 槽函数实现
void FittingChart1::on_btnSavePic_clicked()
{
    QString dir = ModelParameter::instance()->getProjectPath();
    if (dir.isEmpty()) dir = QDir::currentPath();
    QString fileName = QFileDialog::getSaveFileName(this, "保存图片", dir + "/loglog_chart.png", "PNG (*.png);;JPG (*.jpg);;PDF (*.pdf)");
    if (fileName.isEmpty()) return;
    if (fileName.endsWith(".png")) m_plot->savePng(fileName);
    else if (fileName.endsWith(".jpg")) m_plot->saveJpg(fileName);
    else m_plot->savePdf(fileName);
}

void FittingChart1::on_btnExportData_clicked() { emit exportDataTriggered(); }

void FittingChart1::on_btnSetting_clicked()
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

void FittingChart1::on_btnReset_clicked()
{
    m_plot->rescaleAxes();
    // 保护对数坐标轴下限
    if(m_plot->xAxis->range().lower <= 0) m_plot->xAxis->setRangeLower(1e-3);
    if(m_plot->yAxis->range().lower <= 0) m_plot->yAxis->setRangeLower(1e-3);
    m_plot->replot();
}

// 特征线逻辑
void FittingChart1::addCharacteristicLine(double slope)
{
    QCPAxisRect* rect = m_plot->axisRect();
    double lowerX = rect->axis(QCPAxis::atBottom)->range().lower;
    double upperX = rect->axis(QCPAxis::atBottom)->range().upper;
    double lowerY = rect->axis(QCPAxis::atLeft)->range().lower;
    double upperY = rect->axis(QCPAxis::atLeft)->range().upper;

    // Log-Log 下中心点计算
    double centerX = pow(10, (log10(lowerX) + log10(upperX)) / 2.0);
    double centerY = pow(10, (log10(lowerY) + log10(upperY)) / 2.0);

    double x1, y1, x2, y2;
    calculateLinePoints(slope, centerX, centerY, x1, y1, x2, y2);

    QCPItemLine* line = new QCPItemLine(m_plot);
    line->setClipAxisRect(rect);
    line->start->setCoords(x1, y1);
    line->end->setCoords(x2, y2);

    QPen pen(Qt::black, 2, Qt::DashLine);
    line->setPen(pen);
    line->setSelectedPen(QPen(Qt::blue, 2, Qt::SolidLine));
    line->setProperty("fixedSlope", slope);
    line->setProperty("isCharacteristic", true);
    m_plot->replot();
}

void FittingChart1::calculateLinePoints(double slope, double centerX, double centerY, double& x1, double& y1, double& x2, double& y2)
{
    // Log-Log 模式下的点计算
    double span = 3.0; // 跨度
    x1 = centerX / span;
    x2 = centerX * span;
    y1 = centerY * pow(x1 / centerX, slope);
    y2 = centerY * pow(x2 / centerX, slope);
}

// 交互逻辑
void FittingChart1::onPlotMousePress(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton) return;

    m_interMode = Mode_None;
    m_activeLine = nullptr; m_activeText = nullptr; m_activeArrow = nullptr;
    m_lastMousePos = event->pos();
    double tolerance = 8.0;

    // 检查文本选中
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        if (auto text = qobject_cast<QCPItemText*>(m_plot->item(i))) {
            if (text->selectTest(event->pos(), false) < tolerance) {
                m_interMode = Mode_Dragging_Text; m_activeText = text;
                m_plot->deselectAll(); text->setSelected(true);
                m_plot->setInteractions(QCP::Interaction(0));
                m_plot->replot(); return;
            }
        }
    }

    // 检查特征线选中
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        QCPItemLine* line = qobject_cast<QCPItemLine*>(m_plot->item(i));
        if (!line || !line->property("isCharacteristic").isValid()) continue;

        double x1 = m_plot->xAxis->coordToPixel(line->start->coords().x());
        double y1 = m_plot->yAxis->coordToPixel(line->start->coords().y());
        double x2 = m_plot->xAxis->coordToPixel(line->end->coords().x());
        double y2 = m_plot->yAxis->coordToPixel(line->end->coords().y());
        QPointF p(event->pos());

        if (std::sqrt(pow(p.x()-x1,2)+pow(p.y()-y1,2)) < tolerance) { m_interMode=Mode_Dragging_Start; m_activeLine=line; }
        else if (std::sqrt(pow(p.x()-x2,2)+pow(p.y()-y2,2)) < tolerance) { m_interMode=Mode_Dragging_End; m_activeLine=line; }
        else if (distToSegment(p, QPointF(x1,y1), QPointF(x2,y2)) < tolerance) { m_interMode=Mode_Dragging_Line; m_activeLine=line; }

        if (m_interMode != Mode_None) {
            m_plot->deselectAll(); line->setSelected(true);
            m_plot->setInteractions(QCP::Interaction(0));
            m_plot->replot(); return;
        }
    }

    m_plot->deselectAll(); m_plot->replot();
}

void FittingChart1::onPlotMouseMove(QMouseEvent* event)
{
    if (event->buttons() & Qt::LeftButton) {
        QPointF currentPos = event->pos();
        QPointF delta = currentPos - m_lastMousePos;
        double mouseX = m_plot->xAxis->pixelToCoord(currentPos.x());
        double mouseY = m_plot->yAxis->pixelToCoord(currentPos.y());

        if (m_interMode == Mode_Dragging_Text && m_activeText) {
            double px = m_plot->xAxis->coordToPixel(m_activeText->position->coords().x()) + delta.x();
            double py = m_plot->yAxis->coordToPixel(m_activeText->position->coords().y()) + delta.y();
            m_activeText->position->setCoords(m_plot->xAxis->pixelToCoord(px), m_plot->yAxis->pixelToCoord(py));
        }
        else if (m_interMode == Mode_Dragging_Line && m_activeLine) {
            // 平移线段
            double sPx = m_plot->xAxis->coordToPixel(m_activeLine->start->coords().x()) + delta.x();
            double sPy = m_plot->yAxis->coordToPixel(m_activeLine->start->coords().y()) + delta.y();
            double ePx = m_plot->xAxis->coordToPixel(m_activeLine->end->coords().x()) + delta.x();
            double ePy = m_plot->yAxis->coordToPixel(m_activeLine->end->coords().y()) + delta.y();
            m_activeLine->start->setCoords(m_plot->xAxis->pixelToCoord(sPx), m_plot->yAxis->pixelToCoord(sPy));
            m_activeLine->end->setCoords(m_plot->xAxis->pixelToCoord(ePx), m_plot->yAxis->pixelToCoord(ePy));

            updateAnnotationArrow(m_activeLine);
        }
        else if ((m_interMode == Mode_Dragging_Start || m_interMode == Mode_Dragging_End) && m_activeLine) {
            constrainLinePoint(m_activeLine, m_interMode == Mode_Dragging_Start, mouseX, mouseY);
        }

        m_lastMousePos = currentPos; m_plot->replot();
    }
}

void FittingChart1::onPlotMouseRelease(QMouseEvent* event)
{
    Q_UNUSED(event);
    if (m_interMode != Mode_None) {
        m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);
    }
    m_interMode = Mode_None;
}

void FittingChart1::onPlotMouseDoubleClick(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton) return;
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        if (auto text = qobject_cast<QCPItemText*>(m_plot->item(i))) {
            if (text->selectTest(event->pos(), false) < 10.0) { onEditItemRequested(text); return; }
        }
    }
}

// 辅助逻辑
double FittingChart1::distToSegment(const QPointF& p, const QPointF& s, const QPointF& e)
{
    double l2 = (s.x()-e.x())*(s.x()-e.x()) + (s.y()-e.y())*(s.y()-e.y());
    if (l2 == 0) return std::sqrt((p.x()-s.x())*(p.x()-s.x()) + (p.y()-s.y())*(p.y()-s.y()));
    double t = ((p.x()-s.x())*(e.x()-s.x()) + (p.y()-s.y())*(e.y()-s.y())) / l2;
    t = std::max(0.0, std::min(1.0, t));
    QPointF proj = s + t * (e - s);
    return std::sqrt((p.x()-proj.x())*(p.x()-proj.x()) + (p.y()-proj.y())*(p.y()-proj.y()));
}

void FittingChart1::constrainLinePoint(QCPItemLine* line, bool isMovingStart, double mouseX, double mouseY)
{
    double k = line->property("fixedSlope").toDouble();
    double xFixed = isMovingStart ? line->end->coords().x() : line->start->coords().x();
    double yFixed = isMovingStart ? line->end->coords().y() : line->start->coords().y();

    // Log-Log 约束: y = yFixed * (x / xFixed)^k
    if (xFixed <= 0) xFixed = 1e-5;
    if (mouseX <= 0) mouseX = 1e-5;
    double yNew = yFixed * pow(mouseX / xFixed, k);

    if (isMovingStart) line->start->setCoords(mouseX, yNew);
    else line->end->setCoords(mouseX, yNew);
}

void FittingChart1::updateAnnotationArrow(QCPItemLine* line)
{
    if (m_annotations.contains(line)) {
        ChartAnnotation1 note = m_annotations[line];
        double midX = (line->start->coords().x() + line->end->coords().x()) / 2.0;
        double midY = (line->start->coords().y() + line->end->coords().y()) / 2.0;
        if(note.arrowItem) note.arrowItem->end->setCoords(midX, midY);
    }
}

void FittingChart1::onAddAnnotationRequested(QCPItemLine* line) { addAnnotationToLine(line); }
void FittingChart1::onDeleteSelectedRequested() { deleteSelectedItems(); }

void FittingChart1::onLineStyleRequested(QCPItemLine* line) {
    if (!line) return;
    StyleSelectorDialog dlg(StyleSelectorDialog::ModeLine, this);
    dlg.setWindowTitle("样式设置");
    dlg.setPen(line->pen());
    if (dlg.exec() == QDialog::Accepted) {
        line->setPen(dlg.getPen());
        m_plot->replot();
    }
}

void FittingChart1::onEditItemRequested(QCPAbstractItem* item) {
    if (auto text = qobject_cast<QCPItemText*>(item)) {
        bool ok;
        QString newContent = QInputDialog::getText(this, "修改标注", "内容:", QLineEdit::Normal, text->text(), &ok);
        if (ok && !newContent.isEmpty()) { text->setText(newContent); m_plot->replot(); }
    }
}

void FittingChart1::addAnnotationToLine(QCPItemLine* line) {
    if (!line) return;
    if (m_annotations.contains(line)) {
        ChartAnnotation1 old = m_annotations.take(line);
        if(old.textItem) m_plot->removeItem(old.textItem);
        if(old.arrowItem) m_plot->removeItem(old.arrowItem);
    }
    double k = line->property("fixedSlope").toDouble();
    bool ok;
    QString text = QInputDialog::getText(this, "添加标注", "输入:", QLineEdit::Normal, QString("k=%1").arg(k), &ok);
    if (!ok || text.isEmpty()) return;

    QCPItemText* txt = new QCPItemText(m_plot);
    txt->setText(text);
    txt->position->setType(QCPItemPosition::ptPlotCoords);
    double midX = (line->start->coords().x() + line->end->coords().x()) / 2.0;
    double midY = (line->start->coords().y() + line->end->coords().y()) / 2.0;
    txt->position->setCoords(midX, midY * 1.5);

    QCPItemLine* arr = new QCPItemLine(m_plot);
    arr->setHead(QCPLineEnding::esSpikeArrow);
    arr->start->setParentAnchor(txt->bottom);
    arr->end->setCoords(midX, midY);

    ChartAnnotation1 note; note.textItem = txt; note.arrowItem = arr;
    m_annotations.insert(line, note);
    m_plot->replot();
}

void FittingChart1::deleteSelectedItems() {
    auto items = m_plot->selectedItems();
    for (auto item : items) m_plot->removeItem(item);
    m_plot->replot();
}

void FittingChart1::keyPressEvent(QKeyEvent *event) {
    QWidget::keyPressEvent(event);
}

void FittingChart1::closeEvent(QCloseEvent *event)
{
    // 子窗口关闭时提示
    QMessageBox::StandardButton res = QMessageBox::question(
        this, "确认关闭", "确定要隐藏此图表窗口吗？\n(可通过布局恢复显示)",
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (res == QMessageBox::Yes) event->accept(); else event->ignore();
}
