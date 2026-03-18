/*
 * 文件名: chartsetting1.cpp
 * 文件作用: 图表设置对话框实现文件
 * 功能描述:
 * 1. 初始化对话框界面，加载当前图表的各项参数。
 * 2. 实现了曲线列表的自定义显示：
 * - 第一列显示曲线样式预览（线型+点型），只读。
 * - 第二列显示曲线名称，可编辑。
 * - 第三列显示复选框（居中显示），控制曲线在图例中的显示/隐藏。
 * 3. [修复] 增加了对坐标轴指针的空值检查，防止在特殊图表模式下崩溃。
 * 4. [修复] 修正了 QCPPainter 的 drawLine 调用方式，避免重载冲突。
 */

#include "chartsetting1.h"
#include "ui_chartsetting1.h"
#include <QDebug>
#include <QPixmap>
#include <QIcon>
#include <QPainter>
#include <QLineF>
#include <QPointF>
#include <QLabel>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QWidget>

// 构造函数
ChartSetting1::ChartSetting1(MouseZoom* plot, QCPTextElement* title, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ChartSetting1),
    m_plot(plot),
    m_title(title)
{
    ui->setupUi(this);

    // 设置对话框标题
    this->setWindowTitle("图表设置");

    // 设置表格列数和表头
    ui->tableGraphs->setColumnCount(3);
    QStringList headers;
    headers << "样式" << "曲线名称" << "图例显示";
    ui->tableGraphs->setHorizontalHeaderLabels(headers);

    // 设置列宽模式
    ui->tableGraphs->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    ui->tableGraphs->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    ui->tableGraphs->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);

    // 初始化界面数据
    initData();
}

// 析构函数
ChartSetting1::~ChartSetting1()
{
    delete ui;
}

// 初始化数据：从 m_plot 读取当前状态填入 UI
void ChartSetting1::initData()
{
    // [安全检查] 确保 plot 对象有效
    if (!m_plot) return;

    // =========================================================
    // 1. 标题设置
    // =========================================================
    if (m_title) {
        ui->editTitle->setText(m_title->text());
        ui->checkTitleVisible->setChecked(m_title->visible());
    }

    // =========================================================
    // 2. X轴设置
    // =========================================================
    // [安全检查] 确保 xAxis 有效。在某些模式下(如Stacked)，默认Axis可能被隐藏或移除。
    // 如果为空，我们尝试获取第一个可见的 AxisRect 的 X 轴，或者直接返回避免崩溃。
    QCPAxis *x = m_plot->xAxis;
    if (!x && !m_plot->axisRects().isEmpty()) {
        x = m_plot->axisRects().first()->axis(QCPAxis::atBottom);
    }

    if (x) {
        ui->editXLabel->setText(x->label());
        ui->spinXMin->setValue(x->range().lower);
        ui->spinXMax->setValue(x->range().upper);

        bool isLogX = (x->scaleType() == QCPAxis::stLogarithmic);
        ui->checkXLog->setChecked(isLogX);

        QString xFormat = x->numberFormat();
        ui->checkXSci->setChecked(xFormat.contains("e"));

        if(x->grid()) {
            ui->checkXGrid->setChecked(x->grid()->visible());
            ui->checkXSubGrid->setChecked(x->grid()->subGridVisible());
        }
    } else {
        // 如果无法获取有效的 X 轴，禁用相关控件以防误操作
        ui->tabX->setEnabled(false);
    }

    // =========================================================
    // 3. Y轴设置
    // =========================================================
    QCPAxis *y = m_plot->yAxis;
    if (!y && !m_plot->axisRects().isEmpty()) {
        y = m_plot->axisRects().first()->axis(QCPAxis::atLeft);
    }

    if (y) {
        ui->editYLabel->setText(y->label());
        ui->spinYMin->setValue(y->range().lower);
        ui->spinYMax->setValue(y->range().upper);

        bool isLogY = (y->scaleType() == QCPAxis::stLogarithmic);
        ui->checkYLog->setChecked(isLogY);

        QString yFormat = y->numberFormat();
        ui->checkYSci->setChecked(yFormat.contains("e"));

        if(y->grid()) {
            ui->checkYGrid->setChecked(y->grid()->visible());
            ui->checkYSubGrid->setChecked(y->grid()->subGridVisible());
        }
    } else {
        ui->tabY->setEnabled(false);
    }

    // =========================================================
    // 4. 曲线列表初始化 (样式 | 名称 | 图例显示)
    // =========================================================
    int graphCount = m_plot->graphCount();
    ui->tableGraphs->setRowCount(graphCount);

    for(int i = 0; i < graphCount; ++i) {
        QCPGraph *graph = m_plot->graph(i);
        if(!graph) continue;

        // --- 第一列：样式 (图标预览) ---
        QPixmap pix(60, 20);
        pix.fill(Qt::transparent);
        QCPPainter painter(&pix);
        painter.setRenderHint(QPainter::Antialiasing);

        // [加粗] 获取并加粗画笔
        QPen thickPen = graph->pen();
        if (thickPen.style() != Qt::NoPen) {
            thickPen.setWidthF(std::max(thickPen.widthF(), 1.0) + 2.5);
            painter.setPen(thickPen);
            // [修复] 使用 QPointF 明确调用，避免重载歧义
            painter.drawLine(QPointF(0, 10), QPointF(60, 10));
        }

        if (graph->scatterStyle().shape() != QCPScatterStyle::ssNone) {
            QCPScatterStyle scatter = graph->scatterStyle();

            // 如果散点有独立画笔，同样加粗
            if (scatter.pen().style() != Qt::NoPen) {
                QPen sp = scatter.pen();
                sp.setWidthF(std::max(sp.widthF(), 1.0) + 2.5);
                scatter.setPen(sp);
            }

            // 应用加粗画笔并绘制
            // 注意：applyTo 会设置 painter 的 pen 和 brush
            scatter.applyTo(&painter, thickPen);
            scatter.drawShape(&painter, 30, 10);
        }

        // 使用 Widget + Layout + QLabel 实现完全居中
        QWidget *styleWidget = new QWidget();
        QHBoxLayout *styleLayout = new QHBoxLayout(styleWidget);
        styleLayout->setContentsMargins(0, 0, 0, 0);
        styleLayout->setAlignment(Qt::AlignCenter);

        QLabel *styleLabel = new QLabel();
        styleLabel->setPixmap(pix);
        styleLayout->addWidget(styleLabel);

        ui->tableGraphs->setCellWidget(i, 0, styleWidget);

        // --- 第二列：曲线名称 (可编辑) ---
        QTableWidgetItem* nameItem = new QTableWidgetItem(graph->name());
        ui->tableGraphs->setItem(i, 1, nameItem);

        // --- 第三列：是否显示 (复选框，居中) ---
        QWidget *checkWidget = new QWidget();
        QHBoxLayout *checkLayout = new QHBoxLayout(checkWidget);
        checkLayout->setContentsMargins(0, 0, 0, 0);
        checkLayout->setAlignment(Qt::AlignCenter);

        QCheckBox *pCheckBox = new QCheckBox();
        // 检查当前是否在图例中显示
        bool inLegend = (m_plot->legend && m_plot->legend->itemWithPlottable(graph) != nullptr);
        pCheckBox->setChecked(inLegend);

        checkLayout->addWidget(pCheckBox);
        ui->tableGraphs->setCellWidget(i, 2, checkWidget);
    }
}

// 应用设置：将 UI 数据写回 m_plot
void ChartSetting1::applySettings()
{
    if (!m_plot) return;

    // --- 1. 标题应用 ---
    if (m_title) {
        m_title->setText(ui->editTitle->text());
        m_title->setVisible(ui->checkTitleVisible->isChecked());
    }

    // --- 2. X轴应用 ---
    QCPAxis *x = m_plot->xAxis;
    if (!x && !m_plot->axisRects().isEmpty()) x = m_plot->axisRects().first()->axis(QCPAxis::atBottom);

    if (x && ui->tabX->isEnabled()) {
        x->setLabel(ui->editXLabel->text());
        x->setRange(ui->spinXMin->value(), ui->spinXMax->value());

        if (ui->checkXLog->isChecked()) {
            x->setScaleType(QCPAxis::stLogarithmic);
            QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
            x->setTicker(logTicker);
        } else {
            x->setScaleType(QCPAxis::stLinear);
            QSharedPointer<QCPAxisTicker> linTicker(new QCPAxisTicker);
            x->setTicker(linTicker);
        }

        if (ui->checkXSci->isChecked()) {
            x->setNumberFormat("eb");
            x->setNumberPrecision(0);
        } else {
            x->setNumberFormat("g");
            x->setNumberPrecision(5);
        }

        if(x->grid()) {
            x->grid()->setVisible(ui->checkXGrid->isChecked());
            x->grid()->setSubGridVisible(ui->checkXSubGrid->isChecked());
        }
    }

    // --- 3. Y轴应用 ---
    QCPAxis *y = m_plot->yAxis;
    if (!y && !m_plot->axisRects().isEmpty()) y = m_plot->axisRects().first()->axis(QCPAxis::atLeft);

    if (y && ui->tabY->isEnabled()) {
        y->setLabel(ui->editYLabel->text());
        y->setRange(ui->spinYMin->value(), ui->spinYMax->value());

        if (ui->checkYLog->isChecked()) {
            y->setScaleType(QCPAxis::stLogarithmic);
            QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
            y->setTicker(logTicker);
        } else {
            y->setScaleType(QCPAxis::stLinear);
            QSharedPointer<QCPAxisTicker> linTicker(new QCPAxisTicker);
            y->setTicker(linTicker);
        }

        if (ui->checkYSci->isChecked()) {
            y->setNumberFormat("eb");
            y->setNumberPrecision(0);
        } else {
            y->setNumberFormat("g");
            y->setNumberPrecision(5);
        }

        if(y->grid()) {
            y->grid()->setVisible(ui->checkYGrid->isChecked());
            y->grid()->setSubGridVisible(ui->checkYSubGrid->isChecked());
        }
    }

    // --- 4. 曲线列表应用 ---
    int graphCount = m_plot->graphCount();
    if(ui->tableGraphs->rowCount() == graphCount) {
        for(int i = 0; i < graphCount; ++i) {
            QCPGraph *graph = m_plot->graph(i);
            if (!graph) continue;

            // 应用名称修改
            QTableWidgetItem* nameItem = ui->tableGraphs->item(i, 1);
            if(nameItem) {
                QString newName = nameItem->text();
                if(graph->name() != newName) {
                    graph->setName(newName);
                }
            }

            // 应用图例显示修改
            QWidget *checkWidget = ui->tableGraphs->cellWidget(i, 2);
            if (checkWidget) {
                QCheckBox *pCheckBox = checkWidget->findChild<QCheckBox*>();
                if (pCheckBox && m_plot->legend) {
                    bool showLegend = pCheckBox->isChecked();
                    bool currentlyInLegend = (m_plot->legend->itemWithPlottable(graph) != nullptr);

                    if (showLegend && !currentlyInLegend) {
                        graph->addToLegend(m_plot->legend);
                    } else if (!showLegend && currentlyInLegend) {
                        graph->removeFromLegend(m_plot->legend);
                    }
                }
            }
        }
    }

    // 刷新图表
    m_plot->replot();
}

// 确定按钮
void ChartSetting1::on_btnOk_clicked()
{
    applySettings();
    accept();
}

// 应用按钮
void ChartSetting1::on_btnApply_clicked()
{
    applySettings();
}

// 取消按钮
void ChartSetting1::on_btnCancel_clicked()
{
    reject();
}
