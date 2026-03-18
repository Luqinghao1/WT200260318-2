/*
 * 文件名: fittingpage.cpp
 * 文件作用: 拟合页面容器类实现文件
 * 功能描述:
 * 1. 实现了多页签管理逻辑（增删改），集成 FittingNewDialog。
 * 2. 支持创建 FittingWidget (单分析) 和 FittingMultiplesWidget (多分析对比) 两种类型的页签。
 * 3. [修改] 构造函数中设置背景色为白色。
 * 4. [修改] 新建多分析页签时，传递从 Dialog 获取的曲线选择信息。
 */

#include "fittingpage.h"
#include "ui_fittingpage.h"
#include "wt_fittingwidget.h"
#include "fittingnewdialog.h"
#include "modelparameter.h"
#include <QInputDialog>
#include <QMessageBox>
#include <QJsonArray>
#include <QDebug>

// 构造函数
FittingPage::FittingPage(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FittingPage),
    m_modelManager(nullptr)
{
    ui->setupUi(this);

    // [修改] 拟合主界面背景默认为白色
    this->setAttribute(Qt::WA_StyledBackground, true);
    this->setStyleSheet("background-color: white;");
}

FittingPage::~FittingPage()
{
    delete ui;
}

void FittingPage::setModelManager(ModelManager *m)
{
    m_modelManager = m;
    for(int i = 0; i < ui->tabWidget->count(); ++i) {
        QWidget* w = ui->tabWidget->widget(i);
        if(auto fw = qobject_cast<FittingWidget*>(w)) {
            fw->setModelManager(m);
        } else if(auto fmw = qobject_cast<FittingMultiplesWidget*>(w)) {
            fmw->setModelManager(m);
        }
    }
}

void FittingPage::setProjectDataModels(const QMap<QString, QStandardItemModel*> &models)
{
    m_dataMap = models;
    for(int i = 0; i < ui->tabWidget->count(); ++i) {
        QWidget* w = ui->tabWidget->widget(i);
        if(auto fw = qobject_cast<FittingWidget*>(w)) {
            fw->setProjectDataModels(models);
        }
    }
}

void FittingPage::setObservedDataToCurrent(const QVector<double> &t, const QVector<double> &p, const QVector<double> &d)
{
    QWidget* current = ui->tabWidget->currentWidget();
    FittingWidget* fw = qobject_cast<FittingWidget*>(current);

    if (fw) {
        fw->setObservedData(t, p, d);
    } else {
        if(!fw) {
            fw = createNewTab(generateUniqueName("Analysis"));
            fw->setObservedData(t, p, d);
        }
    }
}

void FittingPage::updateBasicParameters()
{
    for(int i = 0; i < ui->tabWidget->count(); ++i) {
        if(auto fw = qobject_cast<FittingWidget*>(ui->tabWidget->widget(i))) {
            fw->updateBasicParameters();
        }
    }
}

FittingWidget* FittingPage::createNewTab(const QString &name, const QJsonObject &initData)
{
    FittingWidget* w = new FittingWidget(this);
    if(m_modelManager) w->setModelManager(m_modelManager);
    w->setProjectDataModels(m_dataMap);
    connect(w, &FittingWidget::sigRequestSave, this, &FittingPage::onChildRequestSave);

    int index = ui->tabWidget->addTab(w, name);
    ui->tabWidget->setCurrentIndex(index);

    if(!initData.isEmpty()) {
        w->loadFittingState(initData);
    }
    return w;
}

// [修改] 创建多分析页签，接收 selections 参数
FittingMultiplesWidget* FittingPage::createNewMultiTab(const QString& name, const QMap<QString, QJsonObject>& states, const QMap<QString, CurveSelection>& selections)
{
    FittingMultiplesWidget* w = new FittingMultiplesWidget(this);

    if(m_modelManager) w->setModelManager(m_modelManager);

    // 传入 states 和 selections
    w->initialize(states, selections);

    int index = ui->tabWidget->addTab(w, name);
    ui->tabWidget->setCurrentIndex(index);

    return w;
}

QString FittingPage::generateUniqueName(const QString &baseName)
{
    QString name = baseName;
    int counter = 1;
    bool exists = true;
    while(exists) {
        exists = false;
        for(int i=0; i<ui->tabWidget->count(); ++i) {
            if(ui->tabWidget->tabText(i) == name) {
                exists = true;
                break;
            }
        }
        if(exists) {
            counter++;
            name = QString("%1 %2").arg(baseName).arg(counter);
        }
    }
    return name;
}

QJsonObject FittingPage::getTabState(int index)
{
    QWidget* w = ui->tabWidget->widget(index);
    if(auto fw = qobject_cast<FittingWidget*>(w)) {
        return fw->getJsonState();
    } else if(auto fmw = qobject_cast<FittingMultiplesWidget*>(w)) {
        return fmw->getJsonState();
    }
    return QJsonObject();
}

void FittingPage::on_btnNewAnalysis_clicked()
{
    QStringList existingNames;
    for(int i=0; i<ui->tabWidget->count(); ++i) {
        existingNames << ui->tabWidget->tabText(i);
    }

    FittingNewDialog dlg(existingNames, this);
    if(dlg.exec() != QDialog::Accepted) return;

    QString newName = dlg.getNewName();
    AnalysisCreateMode mode = dlg.getMode();
    QStringList sources = dlg.getSourceNames();

    if(mode == AnalysisCreateMode::Blank) {
        createNewTab(newName);
    }
    else if(mode == AnalysisCreateMode::CopySingle) {
        if(sources.isEmpty()) return;
        QString sourceName = sources.first();
        for(int i=0; i<ui->tabWidget->count(); ++i) {
            if(ui->tabWidget->tabText(i) == sourceName) {
                QJsonObject state = getTabState(i);
                createNewTab(newName, state);
                break;
            }
        }
    }
    else if(mode == AnalysisCreateMode::CopyMultiple) {
        if(sources.isEmpty()) return;
        QMap<QString, QJsonObject> statesMap;

        // [新增] 获取精细化选择配置
        QMap<QString, CurveSelection> selections = dlg.getSelectionDetails();

        for(const QString& srcName : sources) {
            for(int i=0; i<ui->tabWidget->count(); ++i) {
                if(ui->tabWidget->tabText(i) == srcName) {
                    // 只有单分析页签可以作为源
                    if(qobject_cast<FittingWidget*>(ui->tabWidget->widget(i))) {
                        statesMap.insert(srcName, getTabState(i));
                    }
                    break;
                }
            }
        }

        if(statesMap.isEmpty()) {
            QMessageBox::warning(this, "错误", "未能获取选定分析的数据，请确保源分析是标准的拟合页面。");
            return;
        }

        // 传递 selections
        createNewMultiTab(newName, statesMap, selections);
    }
}

void FittingPage::on_btnRenameAnalysis_clicked()
{
    int idx = ui->tabWidget->currentIndex();
    if(idx < 0) return;
    QString oldName = ui->tabWidget->tabText(idx);
    bool ok;
    QString newName = QInputDialog::getText(this, "重命名", "请输入新的分析名称:", QLineEdit::Normal, oldName, &ok);
    if(ok && !newName.isEmpty()) {
        ui->tabWidget->setTabText(idx, newName);
    }
}

void FittingPage::on_btnDeleteAnalysis_clicked()
{
    int idx = ui->tabWidget->currentIndex();
    if(idx < 0) return;
    if(ui->tabWidget->count() == 1) {
        QMessageBox::warning(this, "警告", "至少需要保留一个分析页面！");
        return;
    }
    if(QMessageBox::question(this, "确认", "确定要删除当前分析页吗？\n此操作不可恢复。") == QMessageBox::Yes) {
        QWidget* w = ui->tabWidget->widget(idx);
        ui->tabWidget->removeTab(idx);
        delete w;
    }
}

void FittingPage::saveAllFittingStates()
{
    QJsonArray analysesArray;
    for(int i=0; i<ui->tabWidget->count(); ++i) {
        QJsonObject pageObj = getTabState(i);
        if(!pageObj.isEmpty()) {
            pageObj["_tabName"] = ui->tabWidget->tabText(i);
            analysesArray.append(pageObj);
        }
    }
    QJsonObject root;
    root["version"] = "2.1";
    root["analyses"] = analysesArray;
    ModelParameter::instance()->saveFittingResult(root);
}

void FittingPage::loadAllFittingStates()
{
    QJsonObject root = ModelParameter::instance()->getFittingResult();
    if(root.isEmpty()) {
        if(ui->tabWidget->count() == 0) createNewTab("Analysis 1");
        return;
    }
    while (ui->tabWidget->count() > 0) {
        QWidget* w = ui->tabWidget->widget(0);
        ui->tabWidget->removeTab(0);
        delete w;
    }

    if(root.contains("analyses") && root["analyses"].isArray()) {
        QJsonArray arr = root["analyses"].toArray();
        for(int i=0; i<arr.size(); ++i) {
            QJsonObject pageObj = arr[i].toObject();
            QString name = pageObj.contains("_tabName") ? pageObj["_tabName"].toString() : QString("Analysis %1").arg(i+1);

            if(pageObj.contains("type") && pageObj["type"].toString() == "multiple") {
                QJsonObject subStates = pageObj["subStates"].toObject();
                QMap<QString, QJsonObject> map;
                for(auto it = subStates.begin(); it != subStates.end(); ++it) {
                    map.insert(it.key(), it.value().toObject());
                }
                // 加载时暂无 selections 信息，默认全选，传空 map 即可
                createNewMultiTab(name, map, QMap<QString, CurveSelection>());
            } else {
                createNewTab(name, pageObj);
            }
        }
    } else {
        createNewTab("Analysis 1", root);
    }
    if(ui->tabWidget->count() == 0) createNewTab("Analysis 1");
}

void FittingPage::onChildRequestSave()
{
    saveAllFittingStates();
    QMessageBox::information(this, "保存成功", "所有分析页的状态已保存到项目文件 (pwt) 中。");
}

void FittingPage::resetAnalysis()
{
    while (ui->tabWidget->count() > 0) {
        QWidget* w = ui->tabWidget->widget(0);
        ui->tabWidget->removeTab(0);
        delete w;
    }
    createNewTab("Analysis 1");
}
