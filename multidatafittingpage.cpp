/**
 * @file multidatafittingpage.cpp
 * @brief 多数据拟合管理页面实现文件
 *
 * 本代码文件的作用与功能：
 * 1. 实现多标签 QTabWidget 容器的动态增、删、改、查。
 * 2. 界面初始化时，自动生成一个默认的空白分析页签。
 * 3. 规范了底层样式表的应用范围，防止背景样式污染子弹窗（修复了加载数据和选择模型弹窗UI错乱的问题）。
 */

#include "multidatafittingpage.h"
#include "ui_multidatafittingpage.h"
#include "wt_multidatafittingwidget.h"

#include <QInputDialog>
#include <QMessageBox>
#include <QDebug>

/**
 * @brief 构造函数：初始化界面，设置局部样式，并打开默认页签
 * @param parent 父级窗口指针
 */
multidatafittingpage::multidatafittingpage(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::multidatafittingpage),
    m_modelManager(nullptr)
{
    ui->setupUi(this);

    // 设置背景色为白色，与拟合界面统一
    this->setAttribute(Qt::WA_StyledBackground, true);
    // 【修改点】将全局样式修改为针对当前类的局部样式，防止样式级联污染 QDialog 弹窗，解决弹窗字体偏大和出现多余横线的问题
    this->setStyleSheet("multidatafittingpage { background-color: white; }");

    // 默认打开一个空的分析页
    on_btnNew_clicked();
}

/**
 * @brief 析构函数：清理内存，释放 UI 资源
 */
multidatafittingpage::~multidatafittingpage()
{
    delete ui;
}

/**
 * @brief 设置模型管理器，并向现有的所有子页签进行传递
 * @param m 试井模型管理器指针
 */
void multidatafittingpage::setModelManager(ModelManager *m)
{
    m_modelManager = m;
    // 遍历所有 Tab 标签，向每个子分析窗口下发 ModelManager
    for (int i = 0; i < ui->tabWidget->count(); ++i) {
        WT_MultidataFittingWidget* w = qobject_cast<WT_MultidataFittingWidget*>(ui->tabWidget->widget(i));
        if (w) w->setModelManager(m);
    }
}

/**
 * @brief 设置项目数据模型映射，向所有子页签同步分发数据
 * @param models 数据模型映射表
 */
void multidatafittingpage::setProjectDataModels(const QMap<QString, QStandardItemModel *> &models)
{
    m_dataMap = models;
    // 遍历所有 Tab 标签，向每个子分析窗口下发数据源
    for (int i = 0; i < ui->tabWidget->count(); ++i) {
        WT_MultidataFittingWidget* w = qobject_cast<WT_MultidataFittingWidget*>(ui->tabWidget->widget(i));
        if (w) w->setProjectDataModels(models);
    }
}

/**
 * @brief 获取当前激活的多数据分析窗口对象
 * @return 当前多数据拟合子窗口的指针，如果没有则返回 nullptr
 */
WT_MultidataFittingWidget* multidatafittingpage::currentWidget()
{
    if (ui->tabWidget->count() == 0) return nullptr;
    return qobject_cast<WT_MultidataFittingWidget*>(ui->tabWidget->currentWidget());
}

/**
 * @brief 槽函数：新建多数据分析页签
 * 在TabWidget中创建一个新的 WT_MultidataFittingWidget 实例并切换至该页
 */
void multidatafittingpage::on_btnNew_clicked()
{
    // 自动生成连续的默认分析名称
    QString name = QString("多数据分析 %1").arg(ui->tabWidget->count() + 1);

    // 创建子部件并下发必要的参数与数据
    WT_MultidataFittingWidget* newWidget = new WT_MultidataFittingWidget(this);
    if (m_modelManager) newWidget->setModelManager(m_modelManager);
    if (!m_dataMap.isEmpty()) newWidget->setProjectDataModels(m_dataMap);

    // 将子部件加入标签页并设为当前激活状态
    int index = ui->tabWidget->addTab(newWidget, name);
    ui->tabWidget->setCurrentIndex(index);
}

/**
 * @brief 槽函数：重命名当前页签
 * 弹出输入框，允许用户修改当前正在查看的标签页的名称
 */
void multidatafittingpage::on_btnRename_clicked()
{
    int index = ui->tabWidget->currentIndex();
    if (index < 0) return;

    bool ok;
    QString oldName = ui->tabWidget->tabText(index);
    // 弹出文字输入对话框获取新名称
    QString newName = QInputDialog::getText(this, "重命名", "请输入新的分析名称:", QLineEdit::Normal, oldName, &ok);

    if (ok && !newName.isEmpty()) {
        ui->tabWidget->setTabText(index, newName);
    }
}

/**
 * @brief 槽函数：删除当前多数据分析页签
 * 检查保留条件并提示用户，确认后删除当前分析标签页释放内存
 */
void multidatafittingpage::on_btnDelete_clicked()
{
    int index = ui->tabWidget->currentIndex();
    if (index < 0) return;

    // 添加最少保留一个页签的判断
    if(ui->tabWidget->count() == 1) {
        QMessageBox::warning(this, "警告", "至少需要保留一个分析页面！");
        return;
    }

    // 二次确认是否删除
    if (QMessageBox::question(this, "确认删除", "确定要删除当前多数据分析页吗？\n删除后数据不可恢复！", QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
        QWidget* w = ui->tabWidget->widget(index);
        ui->tabWidget->removeTab(index);
        w->deleteLater();

        // 容错处理：若所有页签被删完，自动新建一个空页签
        if (ui->tabWidget->count() == 0) {
            on_btnNew_clicked();
        }
    }
}
