/*
 * 文件名: chartwidget.cpp
 * 文件作用: 通用图表组件实现文件
 * 修改记录:
 * 1. [交互优化] 实现了 closeEvent，点击关闭时弹出确认对话框。
 * 2. [界面清理] 移除了底部按钮的初始化代码及 on_btnDrawLine_clicked 槽函数。
 * 3. [功能保持] 保留了 on_btnSavePic_clicked 等函数，供右键菜单信号调用。
 */

#include "chartwidget.h"
#include "ui_chartwidget.h"
#include "chartsetting1.h"
#include "modelparameter.h"
#include "styleselectordialog.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QInputDialog>
#include <cmath>
#include <QDialog>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QColorDialog>
#include <QSpinBox>
#include <QComboBox>

// ============================================================================
// 构造与析构
// ============================================================================

ChartWidget::ChartWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ChartWidget),
    m_dataModel(nullptr),
    m_titleElement(nullptr),
    m_chartMode(Mode_Single),
    m_topRect(nullptr),
    m_bottomRect(nullptr),
    m_interMode(Mode_None),
    m_activeLine(nullptr),
    m_activeText(nullptr),
    m_activeArrow(nullptr),
    m_movingGraph(nullptr)
{
    ui->setupUi(this);
    m_plot = ui->chart;

    // 设置焦点策略，确保能接收键盘事件
    this->setFocusPolicy(Qt::StrongFocus);
    m_plot->setFocusPolicy(Qt::StrongFocus);

    // 初始化界面元素
    initUi();
    // 初始化信号槽连接
    initConnections();
}

ChartWidget::~ChartWidget()
{
    delete ui;
}

// [新增] 关闭事件处理：增加保护弹窗
void ChartWidget::closeEvent(QCloseEvent *event)
{
    QMessageBox::StandardButton res = QMessageBox::question(
        this, "确认关闭", "确定要关闭此图表窗口吗？\n关闭后可以在菜单中重新打开。",
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);

    if (res == QMessageBox::Yes) {
        event->accept();
    } else {
        event->ignore();
    }
}

// ============================================================================
// 初始化与界面配置
// ============================================================================

void ChartWidget::initUi()
{
    // 确保 plotLayout 至少有一行用于放置标题等
    if (m_plot->plotLayout()->rowCount() == 0) m_plot->plotLayout()->insertRow(0);

    // 获取或创建标题元素
    if (m_plot->plotLayout()->elementCount() > 0 && qobject_cast<QCPTextElement*>(m_plot->plotLayout()->element(0, 0))) {
        m_titleElement = qobject_cast<QCPTextElement*>(m_plot->plotLayout()->element(0, 0));
    } else {
        if(m_plot->plotLayout()->element(0,0) != nullptr) {
            m_plot->plotLayout()->insertRow(0);
        }
        m_titleElement = new QCPTextElement(m_plot, "", QFont("Microsoft YaHei", 12, QFont::Bold));
        m_plot->plotLayout()->addElement(0, 0, m_titleElement);
    }

    // 配置默认坐标轴矩形
    setupAxisRect(m_plot->axisRect());

    // 初始化图例设置
    m_plot->legend->setVisible(true);
    QFont legendFont("Microsoft YaHei", 9);
    m_plot->legend->setFont(legendFont);
    m_plot->legend->setBrush(QBrush(QColor(255, 255, 255, 200))); // 半透明背景

    // 确保图例依附于默认 AxisRect
    if (m_plot->axisRect()) {
        m_plot->axisRect()->insetLayout()->addElement(m_plot->legend, Qt::AlignTop | Qt::AlignRight);
    }

    // [注意] 虽然底部按钮已移除，但 m_lineMenu 仍用于逻辑参考，或将来扩展右键菜单
    m_lineMenu = new QMenu(this);
    QAction* actSlope1 = m_lineMenu->addAction("斜率 k = 1 (井筒储集)");
    connect(actSlope1, &QAction::triggered, this, [=](){ addCharacteristicLine(1.0); });

    QAction* actSlopeHalf = m_lineMenu->addAction("斜率 k = 1/2 (线性流)");
    connect(actSlopeHalf, &QAction::triggered, this, [=](){ addCharacteristicLine(0.5); });

    QAction* actSlopeQuarter = m_lineMenu->addAction("斜率 k = 1/4 (双线性流)");
    connect(actSlopeQuarter, &QAction::triggered, this, [=](){ addCharacteristicLine(0.25); });

    QAction* actHorizontal = m_lineMenu->addAction("水平线 (径向流)");
    connect(actHorizontal, &QAction::triggered, this, [=](){ addCharacteristicLine(0.0); });

    // 默认启用水平和垂直缩放/拖拽
    setZoomDragMode(Qt::Horizontal | Qt::Vertical);
}

void ChartWidget::setupAxisRect(QCPAxisRect *rect)
{
    if (!rect) return;
    // 显示顶部坐标轴（作为边框，无刻度文字）
    QCPAxis *topAxis = rect->axis(QCPAxis::atTop);
    topAxis->setVisible(true);
    topAxis->setTickLabels(false);
    connect(rect->axis(QCPAxis::atBottom), SIGNAL(rangeChanged(QCPRange)), topAxis, SLOT(setRange(QCPRange)));

    // 显示右侧坐标轴（作为边框，无刻度文字）
    QCPAxis *rightAxis = rect->axis(QCPAxis::atRight);
    rightAxis->setVisible(true);
    rightAxis->setTickLabels(false);
    connect(rect->axis(QCPAxis::atLeft), SIGNAL(rangeChanged(QCPRange)), rightAxis, SLOT(setRange(QCPRange)));
}

void ChartWidget::initConnections()
{
    // 连接 MouseZoom 提供的自定义信号 (由右键菜单触发)
    connect(m_plot, &MouseZoom::saveImageRequested, this, &ChartWidget::on_btnSavePic_clicked);
    connect(m_plot, &MouseZoom::exportDataRequested, this, &ChartWidget::on_btnExportData_clicked);
    connect(m_plot, &MouseZoom::drawLineRequested, this, &ChartWidget::addCharacteristicLine); // 直接连接绘制函数
    connect(m_plot, &MouseZoom::settingsRequested, this, &ChartWidget::on_btnSetting_clicked);
    connect(m_plot, &MouseZoom::resetViewRequested, this, &ChartWidget::on_btnReset_clicked);

    connect(m_plot, &MouseZoom::addAnnotationRequested, this, &ChartWidget::onAddAnnotationRequested);
    connect(m_plot, &MouseZoom::lineStyleRequested, this, &ChartWidget::onLineStyleRequested);

    connect(m_plot, &MouseZoom::deleteSelectedRequested, this, &ChartWidget::onDeleteSelectedRequested);
    connect(m_plot, &MouseZoom::editItemRequested, this, &ChartWidget::onEditItemRequested);

    // 连接 QCustomPlot 原生鼠标事件
    connect(m_plot, &QCustomPlot::mousePress, this, &ChartWidget::onPlotMousePress);
    connect(m_plot, &QCustomPlot::mouseMove, this, &ChartWidget::onPlotMouseMove);
    connect(m_plot, &QCustomPlot::mouseRelease, this, &ChartWidget::onPlotMouseRelease);
    connect(m_plot, &QCustomPlot::mouseDoubleClick, this, &ChartWidget::onPlotMouseDoubleClick);
}

// ============================================================================
// 标题管理
// ============================================================================

void ChartWidget::setTitle(const QString &title) {
    refreshTitleElement();
    if (m_titleElement) {
        m_titleElement->setText(title);
        m_plot->replot();
    }
}

QString ChartWidget::title() const {
    if (m_titleElement) return m_titleElement->text();
    return QString();
}

void ChartWidget::refreshTitleElement() {
    m_titleElement = nullptr;
    // 尝试在布局中查找标题元素
    if (m_plot->plotLayout()->elementCount() > 0) {
        if (auto el = qobject_cast<QCPTextElement*>(m_plot->plotLayout()->element(0, 0))) {
            m_titleElement = el;
            return;
        }
        for (int i = 0; i < m_plot->plotLayout()->elementCount(); ++i) {
            if (auto el = qobject_cast<QCPTextElement*>(m_plot->plotLayout()->elementAt(i))) {
                m_titleElement = el;
                return;
            }
        }
    }
}

// ============================================================================
// 图表模式与布局核心逻辑
// ============================================================================

MouseZoom *ChartWidget::getPlot() { return m_plot; }
void ChartWidget::setDataModel(QStandardItemModel *model) { m_dataModel = model; }

void ChartWidget::clearGraphs() {
    m_plot->clearGraphs();
    clearEventLines();
    m_plot->replot();
    exitMoveDataMode();
    setZoomDragMode(Qt::Horizontal | Qt::Vertical);
}

void ChartWidget::setChartMode(ChartMode mode) {
    if (m_chartMode == mode) return;
    m_chartMode = mode;

    exitMoveDataMode();

    // 1. 清理现有布局 (保留第一行的标题)
    int rowCount = m_plot->plotLayout()->rowCount();
    for(int i = rowCount - 1; i > 0; --i) {
        m_plot->plotLayout()->removeAt(i);
    }
    m_plot->plotLayout()->simplify();

    // 如果图例在布局中，先移除（稍后根据模式重新添加）
    if (m_plot->legend && m_plot->legend->layout()) {
        m_plot->legend->layout()->take(m_plot->legend);
    }

    if (mode == Mode_Single) {
        // --- 单坐标系模式 ---
        // 恢复默认间距
        m_plot->plotLayout()->setRowSpacing(5);

        QCPAxisRect* defaultRect = new QCPAxisRect(m_plot);
        m_plot->plotLayout()->addElement(1, 0, defaultRect);
        setupAxisRect(defaultRect);
        m_topRect = nullptr;
        m_bottomRect = nullptr;
        setZoomDragMode(Qt::Horizontal | Qt::Vertical);

        // 图例放入唯一坐标系内
        if (defaultRect->insetLayout() && m_plot->legend) {
            defaultRect->insetLayout()->addElement(m_plot->legend, Qt::AlignTop | Qt::AlignRight);
        }

    } else if (mode == Mode_Stacked) {
        // --- 双坐标堆叠模式 (压力+产量) ---
        m_topRect = new QCPAxisRect(m_plot);
        m_bottomRect = new QCPAxisRect(m_plot);

        // [重要] 将行间距设为 0，使上下图表无缝拼接
        m_plot->plotLayout()->setRowSpacing(0);

        // 添加到布局：标题在 0 行，m_topRect 在 1 行，m_bottomRect 在 2 行
        m_plot->plotLayout()->addElement(1, 0, m_topRect);
        m_plot->plotLayout()->addElement(2, 0, m_bottomRect);

        setupAxisRect(m_topRect);
        setupAxisRect(m_bottomRect);

        // [重要] 使用 QCPMarginGroup 强制对齐左右边界
        QCPMarginGroup *marginGroup = new QCPMarginGroup(m_plot);
        m_topRect->setMarginGroup(QCP::msLeft | QCP::msRight, marginGroup);
        m_bottomRect->setMarginGroup(QCP::msLeft | QCP::msRight, marginGroup);

        // [重要] 手动配置边距，消除中间缝隙
        // 上方坐标系：自动计算左、上、右；手动设置底部边距为0
        m_topRect->setAutoMargins(QCP::msLeft | QCP::msTop | QCP::msRight);
        m_topRect->setMargins(QMargins(0, 0, 0, 0));

        // 下方坐标系：自动计算左、下、右；手动设置顶部边距为0
        m_bottomRect->setAutoMargins(QCP::msLeft | QCP::msBottom | QCP::msRight);
        m_bottomRect->setMargins(QMargins(0, 0, 0, 0));

        // [重要] 轴的显隐控制
        // 上方坐标系（压力）：保留X轴刻度线以对齐网格，但隐藏数值标签和轴标题
        QCPAxis* topXAxis = m_topRect->axis(QCPAxis::atBottom);
        topXAxis->setVisible(true);
        topXAxis->setTickLabels(false);
        topXAxis->setLabel("");

        // 下方坐标系（产量）：隐藏顶部的轴，避免重叠
        m_bottomRect->axis(QCPAxis::atTop)->setVisible(false);

        setZoomDragMode(Qt::Horizontal | Qt::Vertical);

        // 设置坐标轴联动：拖动任一坐标系的X轴，另一个随之变化
        connect(m_topRect->axis(QCPAxis::atBottom), SIGNAL(rangeChanged(QCPRange)),
                m_bottomRect->axis(QCPAxis::atBottom), SLOT(setRange(QCPRange)));
        connect(m_bottomRect->axis(QCPAxis::atBottom), SIGNAL(rangeChanged(QCPRange)),
                m_topRect->axis(QCPAxis::atBottom), SLOT(setRange(QCPRange)));

        // 图例通常放在上方图表
        if (m_topRect->insetLayout() && m_plot->legend) {
            m_topRect->insetLayout()->addElement(m_plot->legend, Qt::AlignTop | Qt::AlignRight);
        }
    }

    if (m_plot->legend) m_plot->legend->setVisible(true);
    m_plot->replot();
}

ChartWidget::ChartMode ChartWidget::getChartMode() const { return m_chartMode; }

QCPAxisRect* ChartWidget::getTopRect() {
    if (m_chartMode == Mode_Single) return m_plot->axisRect();
    return m_topRect;
}

QCPAxisRect* ChartWidget::getBottomRect() {
    if (m_chartMode == Mode_Single) return nullptr;
    return m_bottomRect;
}

// ============================================================================
// 开/关井线 (Event Line) 逻辑
// ============================================================================

void ChartWidget::clearEventLines() {
    for (auto line : m_eventLines) {
        if (m_plot->hasItem(line)) {
            m_plot->removeItem(line);
        }
    }
    m_eventLines.clear();
}

void ChartWidget::addEventLine(double x, int type) {
    // type: 0 = 关井 (红色), 1 = 开井 (绿色)
    QColor color = (type == 0) ? Qt::red : Qt::green;
    QPen pen(color);
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);

    // Lambda: 创建并配置线条
    auto createLine = [&](QCPAxisRect* rect) -> QCPItemLine* {
        if (!rect) return nullptr;
        QCPItemLine* line = new QCPItemLine(m_plot);

        // 设置裁剪，确保线条不超出绘图区
        line->setClipAxisRect(rect);
        line->setClipToAxisRect(true);

        line->start->setAxes(rect->axis(QCPAxis::atBottom), rect->axis(QCPAxis::atLeft));
        line->end->setAxes(rect->axis(QCPAxis::atBottom), rect->axis(QCPAxis::atLeft));

        // 设置位置所参考的矩形 (用于 ptAxisRectRatio)
        line->start->setAxisRect(rect);
        line->end->setAxisRect(rect);

        // X轴使用 PlotCoords (实际时间数值)
        line->start->setTypeX(QCPItemPosition::ptPlotCoords);
        line->end->setTypeX(QCPItemPosition::ptPlotCoords);

        // Y轴使用 AxisRectRatio (0=Top, 1=Bottom)
        // 这样可以保证线条在该矩形内垂直贯穿
        line->start->setTypeY(QCPItemPosition::ptAxisRectRatio);
        line->end->setTypeY(QCPItemPosition::ptAxisRectRatio);

        // 设置坐标: 0 为矩形顶部, 1 为矩形底部
        line->start->setCoords(x, 1);
        line->end->setCoords(x, 0);

        line->setPen(pen);
        line->setSelectedPen(QPen(Qt::blue, 2, Qt::DashLine));
        line->setProperty("isEventLine", true);
        line->setLayer("overlay");

        return line;
    };

    QCPItemLine* lineTop = nullptr;
    QCPItemLine* lineBottom = nullptr;

    if (m_chartMode == Mode_Stacked && m_topRect && m_bottomRect) {
        lineTop = createLine(m_topRect);
        lineBottom = createLine(m_bottomRect);

        if (lineTop && lineBottom) {
            QVariant vTop; vTop.setValue(reinterpret_cast<void*>(lineBottom));
            lineTop->setProperty("sibling", vTop);

            QVariant vBot; vBot.setValue(reinterpret_cast<void*>(lineTop));
            lineBottom->setProperty("sibling", vBot);
        }

        if (lineTop) m_eventLines.append(lineTop);
        if (lineBottom) m_eventLines.append(lineBottom);

    } else {
        QCPAxisRect* target = (m_chartMode == Mode_Stacked && m_topRect) ? m_topRect : m_plot->axisRect();
        if(!target) target = m_plot->axisRect();

        lineTop = createLine(target);
        if (lineTop) m_eventLines.append(lineTop);
    }

    m_plot->replot();
}

// ============================================================================
// 工具栏/右键菜单槽函数
// ============================================================================

void ChartWidget::on_btnSavePic_clicked()
{
    QString dir = ModelParameter::instance()->getProjectPath();
    if (dir.isEmpty()) dir = QDir::currentPath();
    QString fileName = QFileDialog::getSaveFileName(this, "保存图片", dir + "/chart_export.png", "PNG (*.png);;JPG (*.jpg);;PDF (*.pdf)");
    if (fileName.isEmpty()) return;
    if (fileName.endsWith(".png")) m_plot->savePng(fileName);
    else if (fileName.endsWith(".jpg")) m_plot->saveJpg(fileName);
    else m_plot->savePdf(fileName);
}

void ChartWidget::on_btnExportData_clicked() { emit exportDataTriggered(); }

void ChartWidget::on_btnSetting_clicked() {
    refreshTitleElement();
    QString oldTitle;
    if (m_titleElement) oldTitle = m_titleElement->text();

    ChartSetting1 dlg(m_plot, m_titleElement, this);

    dlg.exec();

    refreshTitleElement();
    m_plot->replot();

    if (m_titleElement) {
        QString newTitle = m_titleElement->text();
        if (newTitle != oldTitle) {
            emit titleChanged(newTitle);
        }
    }
    emit graphsChanged();
}

void ChartWidget::on_btnReset_clicked() {
    m_plot->rescaleAxes();
    setZoomDragMode(Qt::Horizontal | Qt::Vertical);
    // 避免 Log 坐标系下包含 0
    if(m_plot->xAxis->scaleType()==QCPAxis::stLogarithmic && m_plot->xAxis->range().lower<=0) m_plot->xAxis->setRangeLower(1e-3);
    if(m_plot->yAxis->scaleType()==QCPAxis::stLogarithmic && m_plot->yAxis->range().lower<=0) m_plot->yAxis->setRangeLower(1e-3);
    m_plot->replot();
}

// ============================================================================
// 特征线与标注逻辑
// ============================================================================

void ChartWidget::addCharacteristicLine(double slope) {
    QCPAxisRect* rect = (m_chartMode == Mode_Stacked && m_topRect) ? m_topRect : m_plot->axisRect();
    double lowerX = rect->axis(QCPAxis::atBottom)->range().lower;
    double upperX = rect->axis(QCPAxis::atBottom)->range().upper;
    double lowerY = rect->axis(QCPAxis::atLeft)->range().lower;
    double upperY = rect->axis(QCPAxis::atLeft)->range().upper;

    bool isLogX = (rect->axis(QCPAxis::atBottom)->scaleType() == QCPAxis::stLogarithmic);
    bool isLogY = (rect->axis(QCPAxis::atLeft)->scaleType() == QCPAxis::stLogarithmic);

    // 计算中心点
    double centerX = isLogX ? pow(10, (log10(lowerX) + log10(upperX)) / 2.0) : (lowerX + upperX) / 2.0;
    double centerY = isLogY ? pow(10, (log10(lowerY) + log10(upperY)) / 2.0) : (lowerY + upperY) / 2.0;

    double x1, y1, x2, y2;
    calculateLinePoints(slope, centerX, centerY, x1, y1, x2, y2, isLogX, isLogY);

    QCPItemLine* line = new QCPItemLine(m_plot);
    line->setClipAxisRect(rect);
    line->start->setCoords(x1, y1);
    line->end->setCoords(x2, y2);
    QPen pen(Qt::black, 2, Qt::DashLine);
    line->setPen(pen);
    line->setSelectedPen(QPen(Qt::blue, 2, Qt::SolidLine));
    line->setProperty("fixedSlope", slope);
    line->setProperty("isLogLog", (isLogX && isLogY));
    line->setProperty("isCharacteristic", true);
    m_plot->replot();
}

void ChartWidget::calculateLinePoints(double slope, double centerX, double centerY, double& x1, double& y1, double& x2, double& y2, bool isLogX, bool isLogY) {
    if (isLogX && isLogY) {
        double span = 3.0;
        x1 = centerX / span; x2 = centerX * span;
        y1 = centerY * pow(x1 / centerX, slope); y2 = centerY * pow(x2 / centerX, slope);
    } else {
        QCPAxisRect* rect = m_plot->axisRect();
        x1 = rect->axis(QCPAxis::atBottom)->range().lower;
        x2 = rect->axis(QCPAxis::atBottom)->range().upper;
        y1 = centerY; y2 = centerY;
    }
}

// ============================================================================
// 鼠标交互逻辑 (选中、拖拽、移动)
// ============================================================================

double ChartWidget::distToSegment(const QPointF& p, const QPointF& s, const QPointF& e) {
    double l2 = (s.x()-e.x())*(s.x()-e.x()) + (s.y()-e.y())*(s.y()-e.y());
    if (l2 == 0) return std::sqrt((p.x()-s.x())*(p.x()-s.x()) + (p.y()-s.y())*(p.y()-s.y()));
    double t = ((p.x()-s.x())*(e.x()-s.x()) + (p.y()-s.y())*(e.y()-s.y())) / l2;
    t = std::max(0.0, std::min(1.0, t));
    QPointF proj = s + t * (e - s);
    return std::sqrt((p.x()-proj.x())*(p.x()-proj.x()) + (p.y()-proj.y())*(p.y()-proj.y()));
}

void ChartWidget::onPlotMousePress(QMouseEvent* event) {
    // --- 右键菜单处理 ---
    if (event->button() == Qt::RightButton) {
        // 1. 优先检查是否击中事件线
        for (auto line : m_eventLines) {
            double dist = line->selectTest(event->pos(), false);
            if (dist >= 0 && dist < 10.0) {
                m_activeLine = line; // 临时标记为活动线用于菜单上下文
                QMenu menu(this);
                QAction* actSetting = menu.addAction("开/关井线设置...");
                connect(actSetting, &QAction::triggered, this, &ChartWidget::onEventLineSettingsTriggered);
                menu.exec(event->globalPosition().toPoint());
                return;
            }
        }

        // 2. 双坐标模式下的数据移动菜单
        if (m_chartMode == Mode_Stacked) {
            QMenu menu(this);
            QAction* actMoveX = menu.addAction("数据横向移动 (X Only)");
            connect(actMoveX, &QAction::triggered, this, &ChartWidget::onMoveDataXTriggered);
            QAction* actMoveY = menu.addAction("数据纵向移动 (Y Only)");
            connect(actMoveY, &QAction::triggered, this, &ChartWidget::onMoveDataYTriggered);
            menu.exec(event->globalPosition().toPoint());
            return;
        }
    }

    // --- 左键交互处理 ---
    if (event->button() != Qt::LeftButton) return;

    // 1. 数据移动模式
    if (m_interMode == Mode_Moving_Data_X || m_interMode == Mode_Moving_Data_Y) {
        QCPAxisRect* clickedRect = m_plot->axisRectAt(event->pos());
        if (clickedRect) {
            QList<QCPGraph*> graphs = clickedRect->graphs();
            if (!graphs.isEmpty()) {
                m_movingGraph = graphs.first();
                m_lastMoveDataPos = event->pos();
            } else {
                m_movingGraph = nullptr;
            }
        }
        return;
    }

    // 重置交互状态
    m_interMode = Mode_None; m_activeLine = nullptr; m_activeText = nullptr; m_activeArrow = nullptr; m_lastMousePos = event->pos();
    double tolerance = 8.0;

    // 2. 检查文本选中
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        if (auto text = qobject_cast<QCPItemText*>(m_plot->item(i))) {
            if (text->selectTest(event->pos(), false) < tolerance) {
                m_interMode = Mode_Dragging_Text; m_activeText = text;
                m_plot->deselectAll(); text->setSelected(true); m_plot->setInteractions(QCP::Interaction(0));
                m_plot->replot(); return;
            }
        }
    }

    // 3. 检查事件线选中
    for (auto line : m_eventLines) {
        if (line->selectTest(event->pos(), false) < tolerance) {
            m_plot->deselectAll();
            line->setSelected(true);

            // 联动选中 sibling (上下线段同时变蓝)
            QVariant v = line->property("sibling");
            if (v.isValid()) {
                QCPItemLine* sibling = static_cast<QCPItemLine*>(v.value<void*>());
                if (sibling) sibling->setSelected(true);
            }

            m_interMode = Mode_None; // 事件线不支持拖拽，仅支持选中
            m_plot->replot();
            return;
        }
    }

    // 4. 检查普通线段和箭头选中
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        auto line = qobject_cast<QCPItemLine*>(m_plot->item(i));
        // 排除事件线
        if (line && !line->property("isCharacteristic").isValid() && !line->property("isEventLine").isValid()) {
            double x1 = m_plot->xAxis->coordToPixel(line->start->coords().x()), y1 = m_plot->yAxis->coordToPixel(line->start->coords().y());
            double x2 = m_plot->xAxis->coordToPixel(line->end->coords().x()), y2 = m_plot->yAxis->coordToPixel(line->end->coords().y());
            QPointF p(event->pos());
            if (std::sqrt(pow(p.x()-x1,2)+pow(p.y()-y1,2)) < tolerance) { m_interMode=Mode_Dragging_ArrowStart; m_activeArrow=line; m_plot->setInteractions(QCP::Interaction(0)); return; }
            if (std::sqrt(pow(p.x()-x2,2)+pow(p.y()-y2,2)) < tolerance) { m_interMode=Mode_Dragging_ArrowEnd; m_activeArrow=line; m_plot->setInteractions(QCP::Interaction(0)); return; }
        }
    }

    // 5. 检查特征线选中
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        QCPItemLine* line = qobject_cast<QCPItemLine*>(m_plot->item(i));
        if (!line || !line->property("isCharacteristic").isValid()) continue;
        double x1 = m_plot->xAxis->coordToPixel(line->start->coords().x()), y1 = m_plot->yAxis->coordToPixel(line->start->coords().y());
        double x2 = m_plot->xAxis->coordToPixel(line->end->coords().x()), y2 = m_plot->yAxis->coordToPixel(line->end->coords().y());
        QPointF p(event->pos());
        if (std::sqrt(pow(p.x()-x1,2)+pow(p.y()-y1,2)) < tolerance) { m_interMode=Mode_Dragging_Start; m_activeLine=line; }
        else if (std::sqrt(pow(p.x()-x2,2)+pow(p.y()-y2,2)) < tolerance) { m_interMode=Mode_Dragging_End; m_activeLine=line; }
        else if (distToSegment(p, QPointF(x1,y1), QPointF(x2,y2)) < tolerance) { m_interMode=Mode_Dragging_Line; m_activeLine=line; }

        if (m_interMode != Mode_None) { m_plot->deselectAll(); line->setSelected(true); m_plot->setInteractions(QCP::Interaction(0)); m_plot->replot(); return; }
    }

    // 点击空白处取消选中
    m_plot->deselectAll();
    m_plot->replot();
}

void ChartWidget::onEventLineSettingsTriggered() {
    if (!m_activeLine) return;

    StyleSelectorDialog dlg(StyleSelectorDialog::ModeLine, this);
    dlg.setWindowTitle("开/关井线设置");
    dlg.setPen(m_activeLine->pen());

    if (dlg.exec() == QDialog::Accepted) {
        QPen newPen = dlg.getPen();
        m_activeLine->setPen(newPen);

        // 联动应用到 sibling
        QVariant v = m_activeLine->property("sibling");
        if (v.isValid()) {
            QCPItemLine* sibling = static_cast<QCPItemLine*>(v.value<void*>());
            if (sibling) sibling->setPen(newPen);
        }
        m_plot->replot();
    }
}

void ChartWidget::onPlotMouseMove(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        QPointF currentPos = event->pos();
        QPointF delta = currentPos - m_lastMousePos;
        double mouseX = m_plot->xAxis->pixelToCoord(currentPos.x()), mouseY = m_plot->yAxis->pixelToCoord(currentPos.y());

        // 处理数据移动
        if ((m_interMode == Mode_Moving_Data_X || m_interMode == Mode_Moving_Data_Y) && m_movingGraph) {
            QCPAxis *xAxis = m_movingGraph->keyAxis();
            QCPAxis *yAxis = m_movingGraph->valueAxis();

            double dx = 0;
            double dy = 0;

            if (m_interMode == Mode_Moving_Data_X) {
                dx = xAxis->pixelToCoord(event->pos().x()) - xAxis->pixelToCoord(m_lastMoveDataPos.x());
            } else {
                dy = yAxis->pixelToCoord(event->pos().y()) - yAxis->pixelToCoord(m_lastMoveDataPos.y());
            }

            QSharedPointer<QCPGraphDataContainer> data = m_movingGraph->data();
            for (auto it = data->begin(); it != data->end(); ++it) {
                if (m_interMode == Mode_Moving_Data_X) it->key += dx;
                else it->value += dy;
            }

            // [同步移动开/关井线]
            if (m_interMode == Mode_Moving_Data_X && !m_eventLines.isEmpty()) {
                if (m_chartMode == Mode_Stacked && m_movingGraph->keyAxis()->axisRect() == m_bottomRect) {
                    for(auto line : m_eventLines) {
                        double currentX = line->start->coords().x();
                        double newX = currentX + dx;
                        line->start->setCoords(newX, line->start->coords().y());
                        line->end->setCoords(newX, line->end->coords().y());
                    }
                }
            }

            m_plot->replot();
            m_lastMoveDataPos = event->pos();
            return;
        }

        // 处理拖拽操作
        if (m_interMode == Mode_Dragging_Text && m_activeText) {
            double px = m_plot->xAxis->coordToPixel(m_activeText->position->coords().x()) + delta.x();
            double py = m_plot->yAxis->coordToPixel(m_activeText->position->coords().y()) + delta.y();
            m_activeText->position->setCoords(m_plot->xAxis->pixelToCoord(px), m_plot->yAxis->pixelToCoord(py));
        }
        else if (m_interMode == Mode_Dragging_ArrowStart && m_activeArrow) {
            if(m_activeArrow->start->parentAnchor()) m_activeArrow->start->setParentAnchor(nullptr);
            m_activeArrow->start->setCoords(mouseX, mouseY);
        } else if (m_interMode == Mode_Dragging_ArrowEnd && m_activeArrow) {
            if(m_activeArrow->end->parentAnchor()) m_activeArrow->end->setParentAnchor(nullptr);
            m_activeArrow->end->setCoords(mouseX, mouseY);
        }
        else if (m_interMode == Mode_Dragging_Line && m_activeLine) {
            double sPx = m_plot->xAxis->coordToPixel(m_activeLine->start->coords().x()) + delta.x();
            double sPy = m_plot->yAxis->coordToPixel(m_activeLine->start->coords().y()) + delta.y();
            double ePx = m_plot->xAxis->coordToPixel(m_activeLine->end->coords().x()) + delta.x();
            double ePy = m_plot->yAxis->coordToPixel(m_activeLine->end->coords().y()) + delta.y();
            m_activeLine->start->setCoords(m_plot->xAxis->pixelToCoord(sPx), m_plot->yAxis->pixelToCoord(sPy));
            m_activeLine->end->setCoords(m_plot->xAxis->pixelToCoord(ePx), m_plot->yAxis->pixelToCoord(ePy));

            if (m_annotations.contains(m_activeLine)) {
                auto note = m_annotations[m_activeLine];
                if (note.textItem) {
                    double tPx = m_plot->xAxis->coordToPixel(note.textItem->position->coords().x()) + delta.x();
                    double tPy = m_plot->yAxis->coordToPixel(note.textItem->position->coords().y()) + delta.y();
                    note.textItem->position->setCoords(m_plot->xAxis->pixelToCoord(tPx), m_plot->yAxis->pixelToCoord(tPy));
                }
            }
            updateAnnotationArrow(m_activeLine);
        }
        else if ((m_interMode == Mode_Dragging_Start || m_interMode == Mode_Dragging_End) && m_activeLine) {
            constrainLinePoint(m_activeLine, m_interMode == Mode_Dragging_Start, mouseX, mouseY);
        }

        m_lastMousePos = currentPos; m_plot->replot();
    }
}

void ChartWidget::onPlotMouseRelease(QMouseEvent* event) {
    Q_UNUSED(event);
    if (m_interMode == Mode_Moving_Data_X || m_interMode == Mode_Moving_Data_Y) {
        if (m_movingGraph) {
            emit graphDataModified(m_movingGraph);
        }
        m_movingGraph = nullptr;
    } else {
        if (m_interMode != Mode_None) {
            setZoomDragMode(Qt::Horizontal | Qt::Vertical);
        }
        m_interMode = Mode_None;
    }
}

void ChartWidget::onPlotMouseDoubleClick(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    for (int i = 0; i < m_plot->itemCount(); ++i) {
        if (auto text = qobject_cast<QCPItemText*>(m_plot->item(i))) {
            if (text->selectTest(event->pos(), false) < 10.0) { onEditItemRequested(text); return; }
        }
    }
}

void ChartWidget::onMoveDataXTriggered() {
    m_interMode = Mode_Moving_Data_X;
    m_plot->setInteractions(QCP::Interaction(0));
    m_plot->setCursor(Qt::SizeHorCursor);
    QMessageBox::information(this, "提示", "已进入横向数据移动模式。\n按 ESC 键退出此模式。");
    m_plot->setFocus();
    this->setFocus();
}

void ChartWidget::onMoveDataYTriggered() {
    m_interMode = Mode_Moving_Data_Y;
    m_plot->setInteractions(QCP::Interaction(0));
    m_plot->setCursor(Qt::SizeVerCursor);
    QMessageBox::information(this, "提示", "已进入纵向数据移动模式。\n按 ESC 键退出此模式。");
    m_plot->setFocus();
    this->setFocus();
}

void ChartWidget::onZoomHorizontalTriggered() { setZoomDragMode(Qt::Horizontal); }
void ChartWidget::onZoomVerticalTriggered() { setZoomDragMode(Qt::Vertical); }
void ChartWidget::onZoomDefaultTriggered() { setZoomDragMode(Qt::Horizontal | Qt::Vertical); }

void ChartWidget::setZoomDragMode(Qt::Orientations orientations) {
    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);

    auto configureRect = [&](QCPAxisRect* rect) {
        if(!rect) return;
        rect->setRangeDrag(orientations);
        rect->setRangeZoom(orientations);

        QCPAxis *hAxis = (orientations & Qt::Horizontal) ? rect->axis(QCPAxis::atBottom) : nullptr;
        QCPAxis *vAxis = (orientations & Qt::Vertical) ? rect->axis(QCPAxis::atLeft) : nullptr;

        rect->setRangeDragAxes(hAxis, vAxis);
        rect->setRangeZoomAxes(hAxis, vAxis);
    };

    if (m_chartMode == Mode_Stacked) {
        configureRect(m_topRect);
        configureRect(m_bottomRect);
    } else {
        configureRect(m_plot->axisRect());
    }
}

void ChartWidget::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape && (m_interMode == Mode_Moving_Data_X || m_interMode == Mode_Moving_Data_Y)) {
        exitMoveDataMode();
    }
    QWidget::keyPressEvent(event);
}

void ChartWidget::exitMoveDataMode() {
    if (m_interMode == Mode_Moving_Data_X || m_interMode == Mode_Moving_Data_Y) {
        m_interMode = Mode_None;
        m_movingGraph = nullptr;
        m_plot->setCursor(Qt::ArrowCursor);
        setZoomDragMode(Qt::Horizontal | Qt::Vertical);
    }
}

void ChartWidget::constrainLinePoint(QCPItemLine* line, bool isMovingStart, double mouseX, double mouseY) {
    double k = line->property("fixedSlope").toDouble();
    bool isLogLog = line->property("isLogLog").toBool();
    double xFixed = isMovingStart ? line->end->coords().x() : line->start->coords().x();
    double yFixed = isMovingStart ? line->end->coords().y() : line->start->coords().y();
    double yNew;
    if (isLogLog) {
        if (xFixed <= 0) xFixed = 1e-5;
        if (mouseX <= 0) mouseX = 1e-5;
        yNew = yFixed * pow(mouseX / xFixed, k);
    } else {
        QCPAxisRect* rect = m_plot->axisRect();
        double scale = rect->axis(QCPAxis::atLeft)->range().size() / rect->axis(QCPAxis::atBottom)->range().size();
        yNew = yFixed + (k * scale) * (mouseX - xFixed);
    }
    if (isMovingStart) line->start->setCoords(mouseX, yNew); else line->end->setCoords(mouseX, yNew);
}

void ChartWidget::updateAnnotationArrow(QCPItemLine* line) {
    if (m_annotations.contains(line)) {
        ChartAnnotation note = m_annotations[line];
        double midX = (line->start->coords().x() + line->end->coords().x()) / 2.0;
        double midY = (line->start->coords().y() + line->end->coords().y()) / 2.0;
        if(note.arrowItem) note.arrowItem->end->setCoords(midX, midY);
    }
}

void ChartWidget::onAddAnnotationRequested(QCPItemLine* line) { addAnnotationToLine(line); }
void ChartWidget::onDeleteSelectedRequested() { deleteSelectedItems(); }

void ChartWidget::onLineStyleRequested(QCPItemLine* line) {
    if (!line) return;
    StyleSelectorDialog dlg(StyleSelectorDialog::ModeLine, this);
    dlg.setWindowTitle("标识线样式设置");
    dlg.setPen(line->pen());
    if (dlg.exec() == QDialog::Accepted) {
        line->setPen(dlg.getPen());
        m_plot->replot();
    }
}

void ChartWidget::onEditItemRequested(QCPAbstractItem* item) {
    if (auto text = qobject_cast<QCPItemText*>(item)) {
        bool ok;
        QString newContent = QInputDialog::getText(this, "修改标注", "内容:", QLineEdit::Normal, text->text(), &ok);
        if (ok && !newContent.isEmpty()) { text->setText(newContent); m_plot->replot(); }
    }
}

void ChartWidget::addAnnotationToLine(QCPItemLine* line) {
    if (!line) return;
    if (m_annotations.contains(line)) {
        ChartAnnotation old = m_annotations.take(line);
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

    ChartAnnotation note; note.textItem = txt; note.arrowItem = arr;
    m_annotations.insert(line, note);
    m_plot->replot();
}

void ChartWidget::deleteSelectedItems() {
    auto items = m_plot->selectedItems();
    for (auto item : items) {
        m_plot->removeItem(item);
    }
    m_plot->replot();
}

