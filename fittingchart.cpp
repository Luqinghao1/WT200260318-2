/*
 * 文件名: fittingchart.cpp
 * 文件作用: 拟合绘图管理类实现文件
 * 修改记录:
 * 1. [修复] 在 onPlotMousePress 中修复了无法选中拟合线进行拖拽的Bug (补充了 m_activeLine 赋值)。
 * 2. [新增] 拖拽平移时发送 sigManualPressureUpdated 信号，实现与参数表的联动。
 */

#include "fittingchart.h"
#include <cmath>
#include <algorithm>
#include <QDebug>
#include <QMenu>
#include <QAction>
#include <QMouseEvent>
#include <QMetaObject>
#include <QInputDialog>

FittingChart::FittingChart(QObject *parent)
    : QObject(parent)
    , m_plotLogLog(nullptr), m_plotSemiLog(nullptr), m_plotCartesian(nullptr)
    , m_calculatedPi(0.0)
    , m_isPickingStart(false)
    , m_isPickingEnd(false)
    , m_hasManualPressure(false)
    , m_manualSlope(0)
    , m_manualIntercept(0)
    , m_manualStartX(0)
    , m_manualEndX(0)
    , m_manualTextX(0), m_manualTextY(0)
    , m_interMode(Mode_None)
    , m_activeLine(nullptr)
    , m_activeText(nullptr)
    , m_activeArrow(nullptr)
{
    m_manualTextX = std::numeric_limits<double>::quiet_NaN();
    m_manualTextY = std::numeric_limits<double>::quiet_NaN();
}

// ... (initializeCharts, distToSegment, ContextMenu, Other Slots 保持不变) ...
// 为节省篇幅，此处省略未修改的函数，请保留原有的 initializeCharts 等实现

void FittingChart::initializeCharts(MouseZoom* logLog, MouseZoom* semiLog, MouseZoom* cartesian)
{
    m_plotLogLog = logLog;
    m_plotSemiLog = semiLog;
    m_plotCartesian = cartesian;

    if (m_plotLogLog) {
        m_plotLogLog->xAxis->setLabel("时间 Time (h)");
        m_plotLogLog->yAxis->setLabel("压差 & 导数 (MPa)");
        QSharedPointer<QCPAxisTickerLog> logTickerX(new QCPAxisTickerLog);
        logTickerX->setLogBase(10.0);
        m_plotLogLog->xAxis->setTicker(logTickerX);
        m_plotLogLog->xAxis->setScaleType(QCPAxis::stLogarithmic);
        m_plotLogLog->xAxis->setNumberFormat("gb");
        QSharedPointer<QCPAxisTickerLog> logTickerY(new QCPAxisTickerLog);
        logTickerY->setLogBase(10.0);
        m_plotLogLog->yAxis->setTicker(logTickerY);
        m_plotLogLog->yAxis->setScaleType(QCPAxis::stLogarithmic);
        m_plotLogLog->yAxis->setNumberFormat("gb");
        if(m_plotLogLog->axisRect() && m_plotLogLog->axisRect()->insetLayout())
            m_plotLogLog->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignLeft);
    }

    if (m_plotCartesian) {
        m_plotCartesian->xAxis->setLabel("时间 Time (h)");
        m_plotCartesian->yAxis->setLabel("压差 Delta P (MPa)");
        QSharedPointer<QCPAxisTicker> linearTicker(new QCPAxisTicker);
        m_plotCartesian->xAxis->setTicker(linearTicker);
        m_plotCartesian->xAxis->setScaleType(QCPAxis::stLinear);
        m_plotCartesian->xAxis->setNumberFormat("gb");
        m_plotCartesian->yAxis->setTicker(linearTicker);
        m_plotCartesian->yAxis->setScaleType(QCPAxis::stLinear);
        m_plotCartesian->yAxis->setNumberFormat("gb");
        if(m_plotCartesian->axisRect() && m_plotCartesian->axisRect()->insetLayout())
            m_plotCartesian->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignBottom | Qt::AlignRight);
    }

    if (m_plotSemiLog) {
        if(m_plotSemiLog->axisRect() && m_plotSemiLog->axisRect()->insetLayout())
            m_plotSemiLog->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignRight);
        m_plotSemiLog->disconnect(SIGNAL(customContextMenuRequested(QPoint)));
        m_plotSemiLog->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(m_plotSemiLog, &QWidget::customContextMenuRequested, this, &FittingChart::onSemiLogContextMenu);
        connect(m_plotSemiLog, &QCustomPlot::mousePress, this, &FittingChart::onPlotMousePress);
        connect(m_plotSemiLog, &QCustomPlot::mouseMove, this, &FittingChart::onPlotMouseMove);
        connect(m_plotSemiLog, &QCustomPlot::mouseRelease, this, &FittingChart::onPlotMouseRelease);
        connect(m_plotSemiLog, &QCustomPlot::mouseDoubleClick, [=](QMouseEvent* event){
            if (event->button() != Qt::LeftButton) return;
            for (int i = 0; i < m_plotSemiLog->itemCount(); ++i) {
                if (auto text = qobject_cast<QCPItemText*>(m_plotSemiLog->item(i))) {
                    if (text->selectTest(event->pos(), false) < 10.0) { onEditItemRequested(text); return; }
                }
            }
        });
    }
}

double FittingChart::distToSegment(const QPointF& p, const QPointF& s, const QPointF& e)
{
    double l2 = (s.x()-e.x())*(s.x()-e.x()) + (s.y()-e.y())*(s.y()-e.y());
    if (l2 == 0) return std::sqrt((p.x()-s.x())*(p.x()-s.x()) + (p.y()-s.y())*(p.y()-s.y()));
    double t = ((p.x()-s.x())*(e.x()-s.x()) + (p.y()-s.y())*(e.y()-s.y())) / l2;
    t = std::max(0.0, std::min(1.0, t));
    QPointF proj = s + t * (e - s);
    return std::sqrt((p.x()-proj.x())*(p.x()-proj.x()) + (p.y()-proj.y())*(p.y()-proj.y()));
}

void FittingChart::onSemiLogContextMenu(const QPoint &pos)
{
    if (!m_plotSemiLog) return;
    QMenu *menu = new QMenu(m_plotSemiLog);
    QPointF pMouse = pos;
    double tolerance = 8.0;

    QCPItemLine* hitLine = nullptr;
    QList<QCPItemLine*> candidates;
    if (m_manualFitLine) candidates << m_manualFitLine;
    if (m_manualZeroLine) candidates << m_manualZeroLine;
    for(int i=0; i<m_plotSemiLog->itemCount(); ++i) {
        if(auto l = qobject_cast<QCPItemLine*>(m_plotSemiLog->item(i))) {
            if(!candidates.contains(l) && !l->property("isArrow").toBool()) candidates << l;
        }
    }

    for (auto line : candidates) {
        if (!line->visible()) continue;
        double x1 = m_plotSemiLog->xAxis->coordToPixel(line->start->coords().x());
        double y1 = m_plotSemiLog->yAxis->coordToPixel(line->start->coords().y());
        double x2 = m_plotSemiLog->xAxis->coordToPixel(line->end->coords().x());
        double y2 = m_plotSemiLog->yAxis->coordToPixel(line->end->coords().y());
        double dStart = std::sqrt(pow(pMouse.x()-x1,2) + pow(pMouse.y()-y1,2));
        double dEnd   = std::sqrt(pow(pMouse.x()-x2,2) + pow(pMouse.y()-y2,2));
        double dLine  = distToSegment(pMouse, QPointF(x1, y1), QPointF(x2, y2));
        if (dStart < tolerance || dEnd < tolerance || dLine < tolerance) { hitLine = line; break; }
    }

    QCPItemText* hitText = nullptr;
    for (int i = 0; i < m_plotSemiLog->itemCount(); ++i) {
        if (auto text = qobject_cast<QCPItemText*>(m_plotSemiLog->item(i))) {
            if (text->selectTest(pMouse, false) < tolerance) { hitText = text; break; }
        }
    }

    if (hitLine) {
        m_activeLine = hitLine;
        QAction *actStyle = menu->addAction("线条设置 (颜色/线型)");
        connect(actStyle, &QAction::triggered, [=](){ this->onLineStyleRequested(hitLine); });
        QAction *actAnnotate = menu->addAction("添加/修改标注");
        connect(actAnnotate, &QAction::triggered, [=](){ this->onAddAnnotationRequested(hitLine); });
        menu->addSeparator();
        QAction *actDelete = menu->addAction("删除线段");
        connect(actDelete, &QAction::triggered, this, &FittingChart::onDeleteSelectedRequested);
    }
    else if (hitText) {
        QAction *actEdit = menu->addAction("编辑文本");
        connect(actEdit, &QAction::triggered, [=](){ this->onEditItemRequested(hitText); });
        if (hitText == m_manualResultText) {
            QAction *actResetPos = menu->addAction("重置位置");
            connect(actResetPos, &QAction::triggered, [=](){
                if(m_manualResultText && m_plotSemiLog) {
                    double pixelX = m_plotSemiLog->axisRect()->left() + m_plotSemiLog->axisRect()->width() * 0.05;
                    double pixelY = m_plotSemiLog->axisRect()->top() + m_plotSemiLog->axisRect()->height() * 0.35;
                    m_manualTextX = m_plotSemiLog->xAxis->pixelToCoord(pixelX);
                    m_manualTextY = m_plotSemiLog->yAxis->pixelToCoord(pixelY);
                    m_manualResultText->position->setCoords(m_manualTextX, m_manualTextY);
                    m_plotSemiLog->replot();
                }
            });
        } else {
            QAction *actDelText = menu->addAction("删除文本");
            connect(actDelText, &QAction::triggered, [=](){ m_plotSemiLog->removeItem(hitText); m_plotSemiLog->replot(); });
        }
    }

    if (!hitLine && !hitText) {
        QAction *actSave = menu->addAction("导出图片");
        connect(actSave, &QAction::triggered, [=](){ QMetaObject::invokeMethod(m_plotSemiLog, "saveImageRequested"); });
        QAction *actExport = menu->addAction("导出数据");
        connect(actExport, &QAction::triggered, [=](){ QMetaObject::invokeMethod(m_plotSemiLog, "exportDataRequested"); });
        QMenu* subMenuLine = menu->addMenu("标识线绘制");
        auto addDrawAction = [&](QString name, double k){
            QAction* act = subMenuLine->addAction(name);
            connect(act, &QAction::triggered, [=](){ QMetaObject::invokeMethod(m_plotSemiLog, "drawLineRequested", Q_ARG(double, k)); });
        };
        addDrawAction("斜率 k=1", 1.0);
        addDrawAction("斜率 k=1/2", 0.5);
        addDrawAction("斜率 k=1/4", 0.25);
        addDrawAction("水平线", 0.0);
        QAction *actSetting = menu->addAction("图表设置");
        connect(actSetting, &QAction::triggered, [=](){ QMetaObject::invokeMethod(m_plotSemiLog, "settingsRequested"); });
        menu->addSeparator();
        QAction *actReset = menu->addAction("重置视图");
        connect(actReset, &QAction::triggered, [=](){ QMetaObject::invokeMethod(m_plotSemiLog, "resetViewRequested"); });
        menu->addSeparator();
        QAction *actionSolve = menu->addAction("原始地层压力");
        connect(actionSolve, &QAction::triggered, this, &FittingChart::onShowPressureSolver);
    }
    menu->exec(m_plotSemiLog->mapToGlobal(pos));
    delete menu;
}

void FittingChart::onLineStyleRequested(QCPItemLine* line) {
    if (!line) return;
    StyleSelectorDialog dlg(StyleSelectorDialog::ModeLine, m_plotSemiLog);
    dlg.setPen(line->pen());
    dlg.setWindowTitle("样式设置");
    if (dlg.exec() == QDialog::Accepted) { line->setPen(dlg.getPen()); m_plotSemiLog->replot(); }
}
void FittingChart::onAddAnnotationRequested(QCPItemLine* line) {
    if (!line) return;
    if (m_annotations.contains(line)) {
        FittingChartAnnotation old = m_annotations.take(line);
        if(old.textItem) m_plotSemiLog->removeItem(old.textItem);
        if(old.arrowItem) m_plotSemiLog->removeItem(old.arrowItem);
    }
    bool ok;
    QString defaultText = (line == m_manualFitLine) ? QString("k=%1").arg(m_manualSlope, 0, 'f', 4) : "Annotation";
    QString text = QInputDialog::getText(m_plotSemiLog, "添加标注", "输入标注内容:", QLineEdit::Normal, defaultText, &ok);
    if (!ok || text.isEmpty()) return;
    QCPItemText* txt = new QCPItemText(m_plotSemiLog);
    txt->setText(text);
    txt->position->setType(QCPItemPosition::ptPlotCoords);
    double midX = (line->start->coords().x() + line->end->coords().x()) / 2.0;
    double midY = (line->start->coords().y() + line->end->coords().y()) / 2.0;
    double yOffset = (m_plotSemiLog->yAxis->range().upper - m_plotSemiLog->yAxis->range().lower) * 0.05;
    txt->position->setCoords(midX, midY + yOffset);
    txt->setFont(QFont("Microsoft YaHei", 9));
    QCPItemLine* arr = new QCPItemLine(m_plotSemiLog);
    arr->setHead(QCPLineEnding::esSpikeArrow);
    arr->start->setParentAnchor(txt->bottom);
    arr->end->setCoords(midX, midY);
    arr->setProperty("isArrow", true);
    FittingChartAnnotation note; note.textItem = txt; note.arrowItem = arr;
    m_annotations.insert(line, note);
    m_plotSemiLog->replot();
}
void FittingChart::onDeleteSelectedRequested() {
    if (!m_activeLine) return;
    if (m_activeLine == m_manualFitLine) {
        if (m_manualResultText) { m_plotSemiLog->removeItem(m_manualResultText); m_manualResultText = nullptr; }
        m_hasManualPressure = false;
        m_manualFitLine = nullptr;
    } else if (m_activeLine == m_manualZeroLine) { m_manualZeroLine = nullptr; }
    if (m_annotations.contains(m_activeLine)) {
        FittingChartAnnotation note = m_annotations.take(m_activeLine);
        if(note.textItem) m_plotSemiLog->removeItem(note.textItem);
        if(note.arrowItem) m_plotSemiLog->removeItem(note.arrowItem);
    }
    m_plotSemiLog->removeItem(m_activeLine);
    m_activeLine = nullptr;
    m_plotSemiLog->replot();
}
void FittingChart::onEditItemRequested(QCPAbstractItem* item) {
    if (auto text = qobject_cast<QCPItemText*>(item)) {
        bool ok;
        QString newContent = QInputDialog::getText(m_plotSemiLog, "修改标注", "内容:", QLineEdit::Normal, text->text(), &ok);
        if (ok && !newContent.isEmpty()) { text->setText(newContent); m_plotSemiLog->replot(); }
    }
}
void FittingChart::onShowPressureSolver() {
    if (!m_pressureDialog) {
        m_pressureDialog = new FittingPressureDialog(nullptr);
        connect(m_pressureDialog, &FittingPressureDialog::requestPickStart, this, &FittingChart::onPickStart);
        connect(m_pressureDialog, &FittingPressureDialog::requestPickEnd, this, &FittingChart::onPickEnd);
        connect(m_pressureDialog, &FittingPressureDialog::requestCalculate, this, &FittingChart::onCalculatePressure);
    }
    m_pressureDialog->show(); m_pressureDialog->raise(); m_pressureDialog->activateWindow();
}
void FittingChart::onPickStart() { m_isPickingStart = true; m_isPickingEnd = false; if (m_plotSemiLog) m_plotSemiLog->setCursor(Qt::CrossCursor); }
void FittingChart::onPickEnd() { m_isPickingStart = false; m_isPickingEnd = true; if (m_plotSemiLog) m_plotSemiLog->setCursor(Qt::CrossCursor); }

// ============================================================================
// 鼠标交互核心逻辑
// ============================================================================

void FittingChart::onPlotMousePress(QMouseEvent *event)
{
    if (!m_plotSemiLog) return;

    if (m_isPickingStart || m_isPickingEnd) {
        double x = m_plotSemiLog->xAxis->pixelToCoord(event->pos().x());
        double y = m_plotSemiLog->yAxis->pixelToCoord(event->pos().y());
        if (m_pressureDialog) {
            if (m_isPickingStart) { m_pressureDialog->setStartCoordinate(x, y); m_isPickingStart = false; }
            else if (m_isPickingEnd) { m_pressureDialog->setEndCoordinate(x, y); m_isPickingEnd = false; }
        }
        m_plotSemiLog->setCursor(Qt::ArrowCursor);
        return;
    }

    m_interMode = Mode_None;
    m_activeLine = nullptr; m_activeText = nullptr; m_activeArrow = nullptr;
    m_lastMousePos = event->pos();
    double tolerance = 8.0;

    for (int i = 0; i < m_plotSemiLog->itemCount(); ++i) {
        if (auto text = qobject_cast<QCPItemText*>(m_plotSemiLog->item(i))) {
            if (text->selectTest(event->pos(), false) < tolerance) {
                m_interMode = Mode_Dragging_Text; m_activeText = text;
                m_plotSemiLog->deselectAll(); text->setSelected(true);
                m_plotSemiLog->setInteractions(QCP::Interactions());
                m_plotSemiLog->replot();
                return;
            }
        }
    }

    QList<QCPItemLine*> draggableLines;
    if (m_manualFitLine) draggableLines << m_manualFitLine;
    if (m_manualZeroLine) draggableLines << m_manualZeroLine;
    for(int i=0; i<m_plotSemiLog->itemCount(); ++i) {
        if(auto l = qobject_cast<QCPItemLine*>(m_plotSemiLog->item(i))) {
            if(!draggableLines.contains(l) && !l->property("isArrow").toBool()) draggableLines << l;
        }
    }

    for (auto line : draggableLines) {
        if (!line->visible()) continue;
        double x1 = m_plotSemiLog->xAxis->coordToPixel(line->start->coords().x());
        double y1 = m_plotSemiLog->yAxis->coordToPixel(line->start->coords().y());
        double x2 = m_plotSemiLog->xAxis->coordToPixel(line->end->coords().x());
        double y2 = m_plotSemiLog->yAxis->coordToPixel(line->end->coords().y());
        QPointF p(event->pos());
        double dStart = std::sqrt(pow(p.x()-x1,2) + pow(p.y()-y1,2));
        double dEnd   = std::sqrt(pow(p.x()-x2,2) + pow(p.y()-y2,2));
        double dLine  = distToSegment(p, QPointF(x1, y1), QPointF(x2, y2));

        if (dStart < tolerance) { m_interMode = Mode_Dragging_Start; m_activeLine = line; }
        else if (dEnd < tolerance) { m_interMode = Mode_Dragging_End; m_activeLine = line; }
        else if (dLine < tolerance) {
            // [关键修复]：点击线身时，不仅设置模式，还必须设置 m_activeLine
            if (line == m_manualFitLine) {
                m_interMode = Mode_Dragging_Line;
                m_activeLine = line; // 之前漏了这行
            }
            else m_activeLine = line;
        }

        if (m_activeLine) {
            m_plotSemiLog->deselectAll(); m_activeLine->setSelected(true);
            m_plotSemiLog->setInteractions(QCP::Interactions());
            m_plotSemiLog->replot(); return;
        }
    }
    m_plotSemiLog->deselectAll(); m_plotSemiLog->replot();
}

void FittingChart::onPlotMouseMove(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton) {
        if (!m_plotSemiLog) return;
        QPointF currentPos = event->pos();
        QPointF delta = currentPos - m_lastMousePos;
        double mouseX = m_plotSemiLog->xAxis->pixelToCoord(currentPos.x());
        double mouseY = m_plotSemiLog->yAxis->pixelToCoord(currentPos.y());

        if (m_interMode == Mode_Dragging_Text && m_activeText) {
            if (m_activeText->position->type() == QCPItemPosition::ptPlotCoords) {
                double px = m_plotSemiLog->xAxis->coordToPixel(m_activeText->position->coords().x()) + delta.x();
                double py = m_plotSemiLog->yAxis->coordToPixel(m_activeText->position->coords().y()) + delta.y();
                double newX = m_plotSemiLog->xAxis->pixelToCoord(px);
                double newY = m_plotSemiLog->yAxis->pixelToCoord(py);
                m_activeText->position->setCoords(newX, newY);
                if (m_activeText == m_manualResultText) {
                    m_manualTextX = newX; m_manualTextY = newY;
                }
            }
        }
        else if (m_interMode == Mode_Dragging_Line && m_activeLine == m_manualFitLine) {
            double lastY = m_plotSemiLog->yAxis->pixelToCoord(m_lastMousePos.y());
            double curY = m_plotSemiLog->yAxis->pixelToCoord(currentPos.y());
            double dy = curY - lastY;
            m_manualFitLine->start->setCoords(m_manualFitLine->start->coords().x(), m_manualFitLine->start->coords().y() + dy);
            m_manualFitLine->end->setCoords(m_manualFitLine->end->coords().x(), m_manualFitLine->end->coords().y() + dy);
            m_manualIntercept += dy;
            m_calculatedPi = m_manualIntercept;
            updateManualResultText();
            // [新增] 实时发送信号通知外部更新参数
            emit sigManualPressureUpdated(m_manualSlope, m_manualIntercept);
        }
        else if ((m_interMode == Mode_Dragging_Start || m_interMode == Mode_Dragging_End) && m_activeLine) {
            constrainLinePoint(m_activeLine, m_interMode == Mode_Dragging_Start, mouseX, mouseY);
            updateAnnotationArrow(m_activeLine);
        }
        m_plotSemiLog->replot();
        m_lastMousePos = currentPos;
    }
}

void FittingChart::constrainLinePoint(QCPItemLine* line, bool isMovingStart, double mouseX, double mouseY)
{
    if (line == m_manualZeroLine) {
        if (isMovingStart) line->start->setCoords(0, mouseY); else line->end->setCoords(0, mouseY);
        return;
    }
    if (line == m_manualFitLine) {
        double newY = m_manualSlope * mouseX + m_manualIntercept;
        if (isMovingStart) line->start->setCoords(mouseX, newY); else line->end->setCoords(mouseX, newY);
        return;
    }
    if (isMovingStart) line->start->setCoords(mouseX, mouseY); else line->end->setCoords(mouseX, mouseY);
}

void FittingChart::updateAnnotationArrow(QCPItemLine* line) {
    if (m_annotations.contains(line)) {
        FittingChartAnnotation note = m_annotations[line];
        double midX = (line->start->coords().x() + line->end->coords().x()) / 2.0;
        double midY = (line->start->coords().y() + line->end->coords().y()) / 2.0;
        if(note.arrowItem) note.arrowItem->end->setCoords(midX, midY);
    }
}

void FittingChart::onPlotMouseRelease(QMouseEvent *event)
{
    Q_UNUSED(event);
    if (m_interMode != Mode_None) {
        if (m_plotSemiLog) m_plotSemiLog->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);
    }
    m_interMode = Mode_None;
}

void FittingChart::onCalculatePressure()
{
    if (!m_pressureDialog) return;
    double x1 = m_pressureDialog->getStartX();
    double x2 = m_pressureDialog->getEndX();
    if (x1 > x2) std::swap(x1, x2);
    QVector<double> X, Y;
    double tp = m_settings.producingTime;
    bool useHorner = (tp > 1e-5);
    for(int i=0; i<m_obsT.size(); ++i) {
        double dt = m_obsT[i];
        if (dt < 1e-6) continue;
        double x_val = useHorner ? log10((tp + dt) / dt) : dt;
        if (x_val >= x1 && x_val <= x2) {
            if (i < m_obsRawP.size()) { X.append(x_val); Y.append(m_obsRawP[i]); }
        }
    }
    if (X.size() < 2) return;
    double sumX=0, sumY=0, sumXY=0, sumXX=0;
    int n = X.size();
    for(int i=0; i<n; ++i) { sumX += X[i]; sumY += Y[i]; sumXY += X[i] * Y[i]; sumXX += X[i] * X[i]; }
    double denominator = n * sumXX - sumX * sumX;
    if (std::abs(denominator) < 1e-9) return;
    double k = (n * sumXY - sumX * sumY) / denominator;
    double b = (sumY - k * sumX) / n;
    m_hasManualPressure = true;
    m_manualSlope = k;
    m_manualIntercept = b;
    m_manualStartX = x1;
    m_manualEndX = x2;
    m_calculatedPi = b;

    // 重新计算时，重置文本位置记录
    m_manualTextX = std::numeric_limits<double>::quiet_NaN();
    m_manualTextY = std::numeric_limits<double>::quiet_NaN();

    drawPressureFitResult();

    // [新增] 发送信号更新参数表
    emit sigManualPressureUpdated(m_manualSlope, m_manualIntercept);

    m_plotSemiLog->replot();
}

void FittingChart::drawPressureFitResult()
{
    if (!m_plotSemiLog || !m_hasManualPressure) return;
    if (m_manualFitLine) m_plotSemiLog->removeItem(m_manualFitLine);
    if (m_manualZeroLine) m_plotSemiLog->removeItem(m_manualZeroLine);
    if (m_manualResultText) m_plotSemiLog->removeItem(m_manualResultText);

    double x_start = std::max(m_manualStartX, m_manualEndX);
    if (x_start < m_plotSemiLog->xAxis->range().upper) x_start = m_plotSemiLog->xAxis->range().upper;

    m_manualFitLine = new QCPItemLine(m_plotSemiLog);
    m_manualFitLine->start->setCoords(x_start, m_manualSlope * x_start + m_manualIntercept);
    m_manualFitLine->end->setCoords(0, m_manualIntercept);
    m_manualFitLine->setPen(QPen(Qt::red, 2, Qt::DashLine));
    m_manualFitLine->setSelectable(true);

    m_manualZeroLine = new QCPItemLine(m_plotSemiLog);
    m_manualZeroLine->start->setCoords(0, m_plotSemiLog->yAxis->range().lower);
    m_manualZeroLine->end->setCoords(0, m_plotSemiLog->yAxis->range().upper);
    m_manualZeroLine->setPen(QPen(Qt::blue, 1, Qt::DotLine));
    m_manualZeroLine->setSelectable(true);

    m_manualResultText = new QCPItemText(m_plotSemiLog);
    m_manualResultText->position->setType(QCPItemPosition::ptPlotCoords);

    // 优先使用保存的位置
    if (!std::isnan(m_manualTextX) && !std::isnan(m_manualTextY)) {
        m_manualResultText->position->setCoords(m_manualTextX, m_manualTextY);
    } else {
        double initPixelX = m_plotSemiLog->axisRect()->left() + m_plotSemiLog->axisRect()->width() * 0.05;
        double initPixelY = m_plotSemiLog->axisRect()->top() + m_plotSemiLog->axisRect()->height() * 0.35;
        m_manualTextX = m_plotSemiLog->xAxis->pixelToCoord(initPixelX);
        m_manualTextY = m_plotSemiLog->yAxis->pixelToCoord(initPixelY);
        m_manualResultText->position->setCoords(m_manualTextX, m_manualTextY);
    }

    m_manualResultText->setPositionAlignment(Qt::AlignTop | Qt::AlignLeft);
    m_manualResultText->setFont(QFont("Microsoft YaHei", 9));
    m_manualResultText->setColor(Qt::black);
    m_manualResultText->setBrush(QBrush(QColor(255, 255, 255, 220)));
    m_manualResultText->setPen(QPen(Qt::black));
    m_manualResultText->setPadding(QMargins(4, 4, 4, 4));
    m_manualResultText->setSelectable(true);
    updateManualResultText();
}

void FittingChart::updateManualResultText()
{
    if(!m_manualResultText) return;
    QString slopeStr = QString::number(m_manualSlope, 'f', 4);
    QString interceptStr = QString::number(m_manualIntercept, 'f', 4);
    QString sign = (m_manualIntercept >= 0) ? "+" : "";
    QString text = QString("原始地层压力 P* = %1 MPa\n拟合方程: y = %2x %3 %4")
                       .arg(interceptStr).arg(slopeStr).arg(sign).arg(interceptStr);
    m_manualResultText->setText(text);
}

void FittingChart::setObservedData(const QVector<double>& t, const QVector<double>& deltaP,
                                   const QVector<double>& deriv, const QVector<double>& rawP)
{
    m_obsT = t; m_obsDeltaP = deltaP; m_obsDeriv = deriv; m_obsRawP = rawP;
}

void FittingChart::setSettings(const FittingDataSettings& settings) { m_settings = settings; }
FittingDataSettings FittingChart::getSettings() const { return m_settings; }

void FittingChart::plotAll(const QVector<double>& t_model, const QVector<double>& p_model, const QVector<double>& d_model, bool isModelValid, bool autoScale)
{
    if (!m_plotLogLog || !m_plotSemiLog || !m_plotCartesian) return;
    plotLogLog(t_model, p_model, d_model, isModelValid, autoScale);
    plotSemiLog(t_model, p_model, d_model, isModelValid, autoScale);
    plotCartesian(t_model, p_model, d_model, isModelValid, autoScale);
}

void FittingChart::plotLogLog(const QVector<double>& tm, const QVector<double>& pm, const QVector<double>& dm, bool hasModel, bool autoScale)
{
    MouseZoom* plot = m_plotLogLog;
    plot->clearGraphs();
    plot->clearItems();

    QVector<double> vt, vp, vd;
    for(int i=0; i<m_obsT.size(); ++i) {
        if (m_obsT[i] > 1e-10 && m_obsDeltaP[i] > 1e-10) {
            vt << m_obsT[i]; vp << m_obsDeltaP[i];
            vd << ((i < m_obsDeriv.size() && m_obsDeriv[i] > 1e-10) ? m_obsDeriv[i] : 1e-10);
        }
    }

    plot->addGraph(); plot->graph(0)->setData(vt, vp);
    plot->graph(0)->setPen(Qt::NoPen);
    plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, QColor(0, 100, 0), 6));
    plot->graph(0)->setName("实测压差");

    plot->addGraph(); plot->graph(1)->setData(vt, vd);
    plot->graph(1)->setPen(Qt::NoPen);
    plot->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, Qt::magenta, 6));
    plot->graph(1)->setName("实测导数");

    if (hasModel) {
        QVector<double> vtm, vpm, vdm;
        for(int i=0; i<tm.size(); ++i) {
            if(tm[i] > 1e-10) { vtm << tm[i]; vpm << (pm[i] > 1e-10 ? pm[i] : 1e-10); vdm << (dm[i] > 1e-10 ? dm[i] : 1e-10); }
        }
        plot->addGraph(); plot->graph(2)->setData(vtm, vpm);
        plot->graph(2)->setPen(QPen(Qt::red, 2)); plot->graph(2)->setName("理论压差");
        plot->addGraph(); plot->graph(3)->setData(vtm, vdm);
        plot->graph(3)->setPen(QPen(Qt::blue, 2)); plot->graph(3)->setName("理论导数");
    }

    if (autoScale) {
        plot->rescaleAxes();
        plot->xAxis->scaleRange(1.1, plot->xAxis->range().center());
        plot->yAxis->scaleRange(1.1, plot->yAxis->range().center());
    }
}

void FittingChart::plotSemiLog(const QVector<double>& tm, const QVector<double>& pm, const QVector<double>& dm, bool hasModel, bool autoScale)
{
    MouseZoom* plot = m_plotSemiLog;
    plot->clearGraphs();
    plot->clearItems();
    m_annotations.clear();
    m_manualFitLine = nullptr;
    m_manualZeroLine = nullptr;
    m_manualResultText = nullptr;

    Q_UNUSED(dm);

    double tp = m_settings.producingTime;
    bool useHorner = (tp > 1e-5);

    QVector<double> vt, vp;
    for(int i=0; i<m_obsT.size(); ++i) {
        double dt = m_obsT[i];
        if (dt < 1e-6) continue;
        double x_val = useHorner ? log10((tp + dt) / dt) : dt;
        double y_val = (i < m_obsRawP.size()) ? m_obsRawP[i] : 0.0;
        vt << x_val; vp << y_val;
    }

    plot->addGraph(); plot->graph(0)->setData(vt, vp);
    plot->graph(0)->setPen(Qt::NoPen);
    plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, QColor(0, 0, 180), 5));
    plot->graph(0)->setName("实测压力");

    if (hasModel) {
        QVector<double> vtm, vpm;
        double p_start = !m_obsRawP.isEmpty() ? m_obsRawP.first() : m_settings.initialPressure;
        for(int i=0; i<tm.size(); ++i) {
            double dt = tm[i];
            if (dt < 1e-10) continue;
            double x_val = useHorner ? log10((tp + dt) / dt) : dt;
            double y_val = (m_settings.testType == Test_Drawdown) ? (p_start - pm[i]) : (p_start + pm[i]);
            vtm << x_val; vpm << y_val;
        }
        plot->addGraph(); plot->graph(1)->setData(vtm, vpm);
        plot->graph(1)->setPen(QPen(Qt::red, 2)); plot->graph(1)->setName("理论压力");
    }

    if (m_hasManualPressure) drawPressureFitResult();

    if (useHorner) {
        plot->xAxis->setLabel("Horner 时间比 lg((tp+dt)/dt)");
        plot->yAxis->setLabel("实测压力 Pressure (MPa)");
        QSharedPointer<QCPAxisTicker> linearTicker(new QCPAxisTicker);
        plot->xAxis->setTicker(linearTicker);
        plot->xAxis->setScaleType(QCPAxis::stLinear);
        plot->xAxis->setRangeReversed(false);
        plot->yAxis->setTicker(linearTicker);
        plot->yAxis->setScaleType(QCPAxis::stLinear);
    } else {
        plot->xAxis->setLabel("时间 Time (h)");
        plot->yAxis->setLabel("实测压力 Pressure (MPa)");
        QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
        plot->xAxis->setTicker(logTicker);
        plot->xAxis->setScaleType(QCPAxis::stLogarithmic);
        plot->xAxis->setRangeReversed(false);
        QSharedPointer<QCPAxisTicker> linearTicker(new QCPAxisTicker);
        plot->yAxis->setTicker(linearTicker);
        plot->yAxis->setScaleType(QCPAxis::stLinear);
    }

    if (autoScale) {
        plot->rescaleAxes();
        if(useHorner) {
            plot->xAxis->setRangeUpper(plot->xAxis->range().upper);
            plot->xAxis->setRangeLower(0.0);
        }
    }
}

void FittingChart::plotCartesian(const QVector<double>& tm, const QVector<double>& pm, const QVector<double>& dm, bool hasModel, bool autoScale)
{
    MouseZoom* plot = m_plotCartesian;
    plot->clearGraphs();
    Q_UNUSED(dm);

    QVector<double> vt, vp;
    for(int i=0; i<m_obsT.size(); ++i) { vt << m_obsT[i]; vp << m_obsDeltaP[i]; }

    plot->addGraph();
    plot->graph(0)->setData(vt, vp);
    plot->graph(0)->setPen(Qt::NoPen);
    plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, QColor(0, 100, 0), 6));
    plot->graph(0)->setName("实测压差");

    if (hasModel) {
        QVector<double> vtm, vpm;
        for(int i=0; i<tm.size(); ++i) { vtm << tm[i]; vpm << pm[i]; }
        plot->addGraph();
        plot->graph(1)->setData(vtm, vpm);
        plot->graph(1)->setPen(QPen(Qt::red, 2));
        plot->graph(1)->setName("理论压差");
    }

    if (autoScale) plot->rescaleAxes();
}

void FittingChart::plotSampledPoints(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d)
{
    if (!m_plotLogLog) return;
    QCPGraph* gP = m_plotLogLog->addGraph();
    gP->setData(t, p);
    gP->setPen(Qt::NoPen);
    gP->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, QPen(QColor(0, 100, 0)), QBrush(QColor(0, 100, 0)), 6));
    gP->setName("抽样压差");

    QCPGraph* gD = m_plotLogLog->addGraph();
    gD->setData(t, d);
    gD->setPen(Qt::NoPen);
    gD->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, QPen(Qt::magenta), QBrush(Qt::magenta), 6));
    gD->setName("抽样导数");
}

double FittingChart::getCalculatedInitialPressure() const { return m_calculatedPi; }

QJsonObject FittingChart::getManualPressureState() const
{
    QJsonObject state;
    state["hasManualPressure"] = m_hasManualPressure;
    if (m_hasManualPressure) {
        state["slope"] = m_manualSlope;
        state["intercept"] = m_manualIntercept;
        state["startX"] = m_manualStartX;
        state["endX"] = m_manualEndX;
        state["calculatedPi"] = m_calculatedPi;
        state["textX"] = m_manualTextX;
        state["textY"] = m_manualTextY;
    }
    return state;
}

void FittingChart::setManualPressureState(const QJsonObject& state)
{
    if (state.isEmpty()) return;
    m_hasManualPressure = state["hasManualPressure"].toBool();
    if (m_hasManualPressure) {
        m_manualSlope = state["slope"].toDouble();
        m_manualIntercept = state["intercept"].toDouble();
        m_manualStartX = state["startX"].toDouble();
        m_manualEndX = state["endX"].toDouble();
        m_calculatedPi = state["calculatedPi"].toDouble();
        if (state.contains("textX") && state.contains("textY")) {
            m_manualTextX = state["textX"].toDouble();
            m_manualTextY = state["textY"].toDouble();
        } else {
            m_manualTextX = std::numeric_limits<double>::quiet_NaN();
            m_manualTextY = std::numeric_limits<double>::quiet_NaN();
        }
        drawPressureFitResult();
        if (m_plotSemiLog) m_plotSemiLog->replot();
    }
}
