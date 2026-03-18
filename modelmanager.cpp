/*
 * 文件名: modelmanager.cpp
 * 文件作用: 模型管理类实现
 * 功能描述:
 * 1. 实现了72个模型的初始化管理与界面切换。
 * 2. 实现了计算任务的分发逻辑：Model 1-36 -> Solver1, Model 37-72 -> Solver2。
 * 3. 实现了默认参数的生成逻辑，确保新增加的 Fair/Hegeman 模型参数被正确初始化。
 */

#include "modelmanager.h"
#include "modelselect.h"
#include "modelparameter.h"
#include "wt_modelwidget.h"
#include "modelsolver01-06.h"
#include "modelsolver19_36.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QGroupBox>
#include <QDebug>
#include <cmath>

ModelManager::ModelManager(QWidget* parent)
    : QObject(parent), m_mainWidget(nullptr), m_modelStack(nullptr)
    , m_currentModelType(Model_1)
{
}

ModelManager::~ModelManager()
{
    // 清理求解器内存
    for(auto* s : m_solversGroup1) if(s) delete s;
    m_solversGroup1.clear();
    for(auto* s : m_solversGroup2) if(s) delete s;
    m_solversGroup2.clear();
    // 界面控件由 Qt 父子对象机制管理，无需手动 delete
    m_modelWidgets.clear();
}

void ModelManager::initializeModels(QWidget* parentWidget)
{
    if (!parentWidget) return;
    createMainWidget();

    m_modelStack = new QStackedWidget(m_mainWidget);

    // [扩容] 扩展至 72 个模型槽位
    m_modelWidgets.resize(72);
    m_modelWidgets.fill(nullptr);

    // Group 1: 0-35 (对应 Model 1-1 到 1-36)
    m_solversGroup1.resize(36);
    m_solversGroup1.fill(nullptr);

    // Group 2: 36-71 (对应 Model 2-1 到 2-36)
    m_solversGroup2.resize(36);
    m_solversGroup2.fill(nullptr);

    m_mainWidget->layout()->addWidget(m_modelStack);

    // 默认加载第一个模型
    switchToModel(Model_1);

    if (parentWidget->layout()) parentWidget->layout()->addWidget(m_mainWidget);
    else {
        QVBoxLayout* layout = new QVBoxLayout(parentWidget);
        layout->addWidget(m_mainWidget);
        parentWidget->setLayout(layout);
    }
}

void ModelManager::createMainWidget()
{
    m_mainWidget = new QWidget();
    QVBoxLayout* mainLayout = new QVBoxLayout(m_mainWidget);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);
    m_mainWidget->setLayout(mainLayout);
}

// 懒加载 UI 控件
WT_ModelWidget* ModelManager::ensureWidget(ModelType type)
{
    int index = (int)type;
    if (index < 0 || index >= m_modelWidgets.size()) return nullptr;

    if (m_modelWidgets[index] == nullptr) {
        WT_ModelWidget* widget = new WT_ModelWidget(type, m_modelStack);
        m_modelWidgets[index] = widget;
        m_modelStack->addWidget(widget);

        connect(widget, &WT_ModelWidget::requestModelSelection,
                this, &ModelManager::onSelectModelClicked);
        connect(widget, &WT_ModelWidget::calculationCompleted,
                this, &ModelManager::onWidgetCalculationCompleted);
    }
    return m_modelWidgets[index];
}

// 懒加载 Solver1
ModelSolver01_06* ModelManager::ensureSolverGroup1(int index)
{
    if (index < 0 || index >= m_solversGroup1.size()) return nullptr;
    if (m_solversGroup1[index] == nullptr) {
        m_solversGroup1[index] = new ModelSolver01_06((ModelSolver01_06::ModelType)index);
    }
    return m_solversGroup1[index];
}

// 懒加载 Solver2
ModelSolver19_36* ModelManager::ensureSolverGroup2(int index)
{
    if (index < 0 || index >= m_solversGroup2.size()) return nullptr;
    if (m_solversGroup2[index] == nullptr) {
        m_solversGroup2[index] = new ModelSolver19_36((ModelSolver19_36::ModelType)index);
    }
    return m_solversGroup2[index];
}

void ModelManager::switchToModel(ModelType modelType)
{
    if (!m_modelStack) return;
    ModelType old = m_currentModelType;
    m_currentModelType = modelType;

    WT_ModelWidget* w = ensureWidget(modelType);
    if (w) {
        m_modelStack->setCurrentWidget(w);
    }
    emit modelSwitched(modelType, old);
}

// 核心计算调度
ModelCurveData ModelManager::calculateTheoreticalCurve(ModelType type,
                                                       const QMap<QString, double>& params,
                                                       const QVector<double>& providedTime)
{
    int id = (int)type;
    // [调度] 0-35 分发给 Solver1 (夹层/径向)
    if (id >= 0 && id <= 35) {
        ModelSolver01_06* solver = ensureSolverGroup1(id);
        if (solver) return solver->calculateTheoreticalCurve(params, providedTime);
    }
    // [调度] 36-71 分发给 Solver2 (页岩)
    else if (id >= 36 && id <= 71) {
        ModelSolver19_36* solver = ensureSolverGroup2(id - 36);
        if (solver) return solver->calculateTheoreticalCurve(params, providedTime);
    }
    return ModelCurveData();
}

QString ModelManager::getModelTypeName(ModelType type)
{
    int id = (int)type;
    if (id >= 0 && id <= 35) {
        return ModelSolver01_06::getModelName((ModelSolver01_06::ModelType)id);
    }
    else if (id >= 36 && id <= 71) {
        return ModelSolver19_36::getModelName((ModelSolver19_36::ModelType)(id - 36));
    }
    return "未知模型";
}

void ModelManager::onSelectModelClicked()
{
    ModelSelect dlg(m_mainWidget);
    int currentId = (int)m_currentModelType + 1;

    // 传递 ID (范围 1-72) 给选择对话框回显
    QString currentCode = QString("modelwidget%1").arg(currentId);
    dlg.setCurrentModelCode(currentCode);

    if (dlg.exec() == QDialog::Accepted) {
        QString code = dlg.getSelectedModelCode();
        QString numStr = code;
        numStr.remove("modelwidget");
        bool ok;
        int modelId = numStr.toInt(&ok);
        // ID 范围 1-72
        if (ok && modelId >= 1 && modelId <= 72) {
            switchToModel((ModelType)(modelId - 1));
        }
    }
}

// 生成默认参数 (含新增变井储参数)
QMap<QString, double> ModelManager::getDefaultParameters(ModelType type)
{
    QMap<QString, double> p;
    ModelParameter* mp = ModelParameter::instance();

    // 1. 基础物理参数 (从全局单例获取)
    p.insert("phi", mp->getPhi());
    p.insert("h", mp->getH());
    p.insert("mu", mp->getMu());
    p.insert("B", mp->getB());
    p.insert("Ct", mp->getCt());
    p.insert("q", mp->getQ());
    p.insert("rw", mp->getRw());
    p.insert("L", mp->getL());
    p.insert("nf", mp->getNf());

    // [新增] 变井储参数默认值
    p.insert("alpha", mp->getAlpha());
    p.insert("C_phi", mp->getCPhi());

    // 2. 模型初始猜测值
    p.insert("k", 0.001);
    p.insert("kf", 0.001);
    p.insert("M12", 10.0);
    p.insert("eta12", 1.0);
    p.insert("Lf", 10.0);
    p.insert("rm", 1500.0);
    p.insert("gamaD", 0.006);

    // 3. 根据模型类型决定内/外区参数可见性
    int id = (int)type; // 0-71

    // Group 1 (Solver 1):
    //   0-11: Inner Dual + Outer Dual
    //   12-23: Inner Dual + Outer Homo
    //   24-35: Inner Homo + Outer Homo
    // Group 2 (Solver 2):
    //   36-47: Inner Shale + Outer Shale
    //   48-59: Inner Shale + Outer Homo
    //   60-71: Inner Shale + Outer Dual

    bool hasInnerParams = false;
    bool hasOuterParams = false;

    if (id <= 35) { // Solver 1
        if (id <= 23) hasInnerParams = true; // 0-23 Inner Dual
        if (id <= 11) hasOuterParams = true; // 0-11 Outer Dual
    } else { // Solver 2
        // Inner is always Shale -> Need params (omega, lambda)
        hasInnerParams = true;
        int sub = id - 36;
        // Outer:
        // 0-11 (36-47): Shale -> Need params
        // 12-23 (48-59): Homo -> No params
        // 24-35 (60-71): Dual -> Need params
        if (sub <= 11 || sub >= 24) hasOuterParams = true;
    }

    if (hasInnerParams) {
        p.insert("omega1", 0.4);
        p.insert("lambda1", 0.001);
    }
    if (hasOuterParams) {
        p.insert("omega2", 0.08);
        p.insert("lambda2", 0.0001);
    }

    // 4. 井储与表皮
    // 4种循环: 0:Const, 1:Line, 2:Fair, 3:Hegeman
    int storageType = id % 4;

    // 线源解(1)不需要 C 和 S，其他(0,2,3)都需要
    bool hasStorage = (storageType != 1);

    if (hasStorage) {
        p.insert("cD", 10.0);
        p.insert("S", 0.1);
    } else {
        p.insert("cD", 0.0);
        p.insert("S", 0.0);
    }

    // 5. 边界半径 re
    // 12个一组，前4无限大(不需要re)，后8需要
    int groupIdx = id % 12;
    bool isInfinite = (groupIdx < 4);
    if (!isInfinite) {
        p.insert("re", 20000.0);
    }

    return p;
}

// --- 以下为缓存和工具函数，保持原样 ---
void ModelManager::setHighPrecision(bool high) {
    for(WT_ModelWidget* w : m_modelWidgets) if(w) w->setHighPrecision(high);
    for(ModelSolver01_06* s : m_solversGroup1) if(s) s->setHighPrecision(high);
    for(ModelSolver19_36* s : m_solversGroup2) if(s) s->setHighPrecision(high);
}

void ModelManager::updateAllModelsBasicParameters() {
    for(WT_ModelWidget* w : m_modelWidgets) {
        if(w) QMetaObject::invokeMethod(w, "onResetParameters");
    }
}

void ModelManager::setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d) {
    m_cachedObsTime = t; m_cachedObsPressure = p; m_cachedObsDerivative = d;
}
void ModelManager::getObservedData(QVector<double>& t, QVector<double>& p, QVector<double>& d) const {
    t = m_cachedObsTime; p = m_cachedObsPressure; d = m_cachedObsDerivative;
}
void ModelManager::clearCache() {
    m_cachedObsTime.clear(); m_cachedObsPressure.clear(); m_cachedObsDerivative.clear();
}
bool ModelManager::hasObservedData() const { return !m_cachedObsTime.isEmpty(); }
void ModelManager::onWidgetCalculationCompleted(const QString &t, const QMap<QString, double> &r) {
    emit calculationCompleted(t, r);
}
QVector<double> ModelManager::generateLogTimeSteps(int count, double startExp, double endExp) {
    return ModelSolver01_06::generateLogTimeSteps(count, startExp, endExp);
}
