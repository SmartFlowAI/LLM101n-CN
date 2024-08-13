#include <iostream>
#include <cmath>
#include <set>
#include <vector>
#include <cassert>
#include <tuple>
#include <functional>
#include <algorithm>
#include <numeric>
#include "math.h"
#include <ctime>
#include <random>

class RNG {
// private:
//     uint64_t state;
public:
    RNG(uint64_t seed) : state(seed) {}
    std::default_random_engine e;

    float uniform(float a = 0.0f, float b = 1.0f) {
        std::uniform_real_distribution<float> rand(a, b);
        e.seed(time(0));
        // 使用默认的随机设备创建种子
        std::random_device rd;

        // 使用种子初始化梅森旋转引擎
        std::mt19937 mt(rd());
        return rand(mt);
    }

private:
    uint64_t state;
};

std::tuple<std::vector<std::pair<std::vector<double>, int>>, 
           std::vector<std::pair<std::vector<double>, int>>, 
           std::vector<std::pair<std::vector<double>, int>>>
gen_data(RNG &random, int n = 100) {
    std::vector<std::pair<std::vector<double>, int>> pts;
    for (int i = 0; i < n; ++i) {
        float x = random.uniform(-2.0f, 2.0f);
        float y = random.uniform(-2.0f, 2.0f);
        int label = (x < 0) ? 0 : (y < 0) ? 1 : 2;
        pts.emplace_back(std::vector<double>{x, y}, label);
    }

    // create train/val/test splits of the data (80%, 10%, 10%)
    int tr_size = static_cast<int>(0.8 * n);
    int val_size = static_cast<int>(0.1 * n);

    std::vector<std::pair<std::vector<double>, int>> tr(pts.begin(), pts.begin() + tr_size);
    std::vector<std::pair<std::vector<double>, int>> val(pts.begin() + tr_size, pts.begin() + tr_size + val_size);
    std::vector<std::pair<std::vector<double>, int>> te(pts.begin() + tr_size + val_size, pts.end());

    return std::make_tuple(tr, val, te);
}

// Value class for automatic differentiation
class Value {
public:
    double m;
    double v;
    double data;
    double grad;
    std::set<Value*> prev;
    std::string op;
    std::function<void()> backward;

public:
    Value(double data, std::initializer_list<Value*> children = {}, const std::string& op = "")
        : data(data), grad(0), op(op) {
        for (auto child : children) {
            prev.insert(child);
        }
        backward = [this]() { }; // Default backward function does nothing
    }

    Value(const Value& other): data(other.data), grad(other.grad), prev(other.prev), op(other.op), backward(other.backward){

    }
    Value pow(double other) {
        assert(("Supporting only int/float powers for now", std::floor(other) == other));
        Value out(std::pow(data, other), {this}, "**" + std::to_string(static_cast<int>(other)));
        out.setBackward([this, other, &out]() {
            this->addGrad(other * std::pow(this->data, other - 1) * out.getGrad());
        });
        return out;
    }

    Value relu() {
        Value out(data < 0 ? 0 : data, {this}, "ReLU");
        out.setBackward([this, &out]() {
            this->addGrad((out.getData() > 0) * out.getGrad());
        });
        return out;
    }

    Value tanh() {
        Value out(std::tanh(data), {this}, "tanh");
        out.setBackward([this, &out]() {
            this->addGrad((1 - std::pow(out.getData(), 2)) * out.getGrad());
        });
        return out;
    }

    Value exp() {
        Value out(std::exp(data), {this}, "exp");
        out.setBackward([this, &out]() {
            this->addGrad(std::exp(this->data) * out.getGrad());
        });
        return out;
    }

    Value log() {
        Value out(std::log(data), {this}, "log");
        out.setBackward([this, &out]() {
            this->addGrad(1 / this->data * out.getGrad());
        });
        return out;
    }

    void backwardPass() {
        std::vector<Value*> topo;
        std::set<Value*> visited;

        std::function<void(Value*)> buildTopo = [&topo, &visited, &buildTopo](Value* v) {
            if (visited.insert(v).second) {
                for (auto child : v->prev) {
                    buildTopo(child);
                }
                topo.push_back(v);
            }
        };

        buildTopo(this);

        this->grad = 1; // seed the gradient
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }
    }

    void setBackward(const std::function<void()>& func) {
        backward = func;
    }

    double getData() const {
        return data;
    }

    double getGrad() const {
        return grad;
    }

    void setData(double d) {
        data = d;
    }

    void addGrad(double g) {
        grad += g;
    }

    Value operator+(Value const& other) {
        Value out(data + other.data, {this, &const_cast<Value&>(other)}, "+");
        out.setBackward([this, &out, &other]() {
            this->addGrad(out.getGrad());
            const_cast<Value&>(other).addGrad(out.getGrad());
        });
        return out;
    }

    Value operator*(Value const& other) {
        Value out(data * other.data, {this, &const_cast<Value&>(other)}, "*");
        out.setBackward([this, &out, &other]() {
            this->addGrad(other.data * out.getGrad());
            const_cast<Value&>(other).addGrad(this->data * out.getGrad());
        });
        return out;
    }

    Value operator-(Value const& other) {
        Value out(this->data - other.data, {this, &const_cast<Value&>(other)}, "-");
        out.setBackward([this, &out, &other]() mutable {
            this->addGrad(1.0 * out.getGrad());
            const_cast<Value&>(other).addGrad(-1.0 * out.getGrad());
        });
        return out;
    }

    Value operator/(Value const& other) {
        assert(other.data != 0 && "Division by zero!");
        Value out(this->data / other.data, {this, &const_cast<Value&>(other)}, "/");
        out.setBackward([this, &out, &other]() mutable {
            this->addGrad((1.0 / other.data) * out.getGrad());
            const_cast<Value&>(other).addGrad(-(this->data / (other.data * other.data)) * out.getGrad());
        });
        return out;
    }

    Value operator-() {
        return (*this) * -1;
    }

    Value& operator+=(const Value& other) {
        // 创建this对象的副本，保存当前的状态
        Value* originalThis = new Value(*this);
        // 更新当前对象的数据
        this->data += other.data;
        this->op = "+=";
        // 更新prev集合，保存this对象的原始状态
        this->prev.insert(originalThis);
        this->prev.insert(&const_cast<Value&>(other));
        // 设置backward函数，以便正确计算梯度
        this->backward = [this, originalThis, &other]() {
            originalThis->grad += this->grad;  // 传递梯度给原始的this对象
            const_cast<Value&>(other).grad += this->grad;          // 传递梯度给other对象
        };
        return *this;
    }

    Value& operator-=(const Value& other) {
        // 创建this对象的副本，保存当前的状态
        Value* originalThis = new Value(*this);
        // 更新当前对象的数据
        this->data -= other.data;
        this->op = "-=";
        // 更新prev集合，保存this对象的原始状态
        this->prev.insert(originalThis);
        this->prev.insert(&const_cast<Value&>(other));
        // 设置backward函数，以便正确计算梯度
        this->backward = [this, originalThis, &other]() {
            originalThis->grad += this->grad;  // 传递梯度给原始的this对象
            const_cast<Value&>(other).grad -= this->grad;          // 传递负的梯度给other对象
        };
        return *this;
    }
    
    Value& operator*=(const Value& other) {
        // 创建this对象的副本，保存当前的状态
        Value* originalThis = new Value(*this);
        // 更新当前对象的数据
        this->data *= other.data;
        this->op = "*=";
        // 更新prev集合，保存this对象的原始状态
        this->prev.insert(originalThis);
        this->prev.insert(&const_cast<Value&>(other));
        // 设置backward函数，以便正确计算梯度
        this->backward = [this, originalThis, &other]() {
            originalThis->grad += other.data * this->grad;  // 传递乘积的梯度给原始的this对象
            const_cast<Value&>(other).grad += originalThis->data * this->grad;  // 传递乘积的梯度给other对象
        };
        return *this;
    }

    Value& operator/=(const Value& other) {
        // 创建this对象的副本，保存当前的状态
        Value* originalThis = new Value(*this);
        // 更新当前对象的数据
        this->data /= other.data;
        this->op = "/=";
        // 更新prev集合，保存this对象的原始状态
        this->prev.insert(originalThis);
        this->prev.insert(&const_cast<Value&>(other));
        // 设置backward函数，以便正确计算梯度
        this->backward = [this, originalThis, &other]() {
            originalThis->grad += (1 / other.data) * this->grad;  // 传递除法的梯度给原始的this对象
            const_cast<Value&>(other).grad -= (originalThis->data / (other.data * other.data)) * this->grad;  // 传递除法的梯度给other对象
        };
        return *this;
    }
    
    bool operator<(Value& other){
        return (*this).getData() < other.getData();
    }
    bool operator>(Value& other){
        return (*this).getData() > other.getData();
    }

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        return os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
    }
};


// Base Module class
class Module {
public:
    virtual std::vector<Value*> parameters() = 0;
    virtual ~Module() {}

    void zero_grad() {
        for (auto p : parameters()) {
            p->grad = 0;
        }
    }
};

// Neuron class
class Neuron : public Module {
private:
    std::vector<Value> w;
    Value b;
    bool nonlin;

public:
    Neuron(int nin, bool nonlin = true) : b(0), nonlin(nonlin) {
        RNG random(42);
        for (int i = 0; i < nin; i++) {
            w.emplace_back(Value(random.uniform(-1,1) / sqrt(nin)));
        }
    }

    Value operator()(const std::vector<Value> x) {
        RNG random(42);
        Value act = b;
        for (size_t i = 0; i < w.size(); ++i) {
            act = act + w[i] * x[i];
        }
        return nonlin ? act.tanh() : act;
    }

    std::vector<Value*> parameters() override {
        std::vector<Value*> params;
        for (auto& weight : w) {
            params.push_back(&weight);
        }
        params.push_back(&b);
        return params;
    }
};

// Layer class
class Layer : public Module {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int nin, int nout, bool nonlin = true) {
        for (int i = 0; i < nout; i++) {
            neurons.emplace_back(nin, nonlin);
        }
    }

    std::vector<Value> operator()(const std::vector<Value> x) {
        std::vector<Value> out;
        for (auto& neuron : neurons) {
            out.push_back(neuron(x));
        }
        return out;
    }

    std::vector<Value*> parameters() override {
        std::vector<Value*> params;
        for (auto& neuron : neurons) {
            auto neuronParams = neuron.parameters();
            params.insert(params.end(), neuronParams.begin(), neuronParams.end());
        }
        return params;
    }
};

// MLP class
class MLP : public Module {
private:
    std::vector<Layer> layers;

public:
    MLP(int nin, const std::vector<int>& nouts) {
        int size = nouts.size();
        for (int i = 0; i < size; ++i) {
            layers.emplace_back(Layer(nin, nouts[i], i != size - 1));
            nin = nouts[i];
        }
    }

    std::vector<Value> operator()(std::vector<Value> x) {
        for (auto& layer : layers) {
            x = layer(x);
        }
        return x;
    }

    std::vector<Value*> parameters() override {
        std::vector<Value*> params;
        for (auto& layer : layers) {
            auto layerParams = layer.parameters();
            params.insert(params.end(), layerParams.begin(), layerParams.end());
        }
        return params;
    }
};

Value cross_entropy(std::vector<Value>& logits, int target) {
    // Subtract the max for numerical stability (avoids overflow)
    Value max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<Value> adjusted_logits;
    adjusted_logits.reserve(logits.size());
    
    // Subtract max_val from each logit to prevent overflow
    for (auto val : logits) {
        adjusted_logits.push_back(val - max_val);
    }

    // 1) Evaluate elementwise e^x
    std::vector<Value> ex;
    ex.reserve(adjusted_logits.size());
    for (auto val : adjusted_logits) {
        ex.push_back(val.exp());
    }

    // 2) Compute the sum of the above
    Value denom = std::accumulate(ex.begin(), ex.end(), Value(0.0));

    // 3) Normalize by the sum to get probabilities
    std::vector<Value> probs;
    probs.reserve(ex.size());
    for (auto val : ex) {
        probs.push_back(val / denom);
    }

    // 4) Log the probabilities at the target index
    Value logp = probs[target].log();

    // 5) The negative log likelihood loss (invert so we get a loss - lower is better)
    Value nll = -logp;

    return nll;
}


// 评估数据集 split 的损失
double eval_split(MLP& model, const std::vector<std::pair<std::vector<double>, int>>& split) {
    Value loss(0);
    for (const auto& sample : split) {
        const std::vector<double>& x = sample.first;
        int y = sample.second;
        
        std::vector<Value> inputs = { Value(x[0]), Value(x[1]) };
        std::vector<Value> logits = model(inputs);
        Value cer = cross_entropy(logits, y);
        loss += cer;
    }
    loss = loss * (1.0 / split.size()); // 归一化损失
    return loss.getData();
}


int main() {
    // 数据生成
    RNG random(42);
    auto datasets = gen_data(random, 100);
    std::vector<std::pair<std::vector<double>, int>> train_split = std::get<0>(datasets);  // 数据集存储
    std::vector<std::pair<std::vector<double>, int>> val_split  = std::get<1>(datasets); // 数据集存储
    std::vector<std::pair<std::vector<double>, int>> test_split  = std::get<2>(datasets);
    
    
    MLP model(2, {16, 3});

    double learning_rate = 0.5;
    double beta1 = 0.9;
    double beta2 = 0.95;
    double weight_decay = 0.0001;

    // 参数初始化
    for (auto* p : model.parameters()) {
        p->grad = 0.0;  // 初始化梯度
        p->m = 0.0;
        p->v = 0.0;
    }

    for (int step = 0; step < 2000; ++step) {
        if (step % 10 == 0) {
            // 验证集评估
            double val_loss = eval_split(model, val_split);
            std::cout << "step " << step << ", val loss " << val_loss << std::endl;
        }

        Value loss(0.0);
        for (auto& data : train_split) {
            std::vector<Value> x = {Value(data.first[0]), Value(data.first[1])};
            std::vector<Value> logits = model(x);
            
            Value ce(cross_entropy(logits, data.second));
            loss += ce;
        }
        // Value train_size(train_split.size());
        loss = loss * (1.0/train_split.size());

        loss.backwardPass();

        // 参数更新（AdamW）
        for (auto* p : model.parameters()) {
            // 梯度更新逻辑
            p->m = beta1 * p->m + (1 - beta1) * p->grad;
            p->v = beta2 * p->v + (1 - beta2) * p->grad * p->grad;
            double m_hat = p->m / (1 - pow(beta1, step + 1));  // 偏差修正
            double v_hat = p->v / (1 - pow(beta2, step + 1));
            p->data -= learning_rate * (m_hat / (sqrt(v_hat) + 1e-8) + weight_decay * p->data);
        }

        model.zero_grad();

        std::cout << "Step " << step << ", Train Loss: " << loss << std::endl;
    }

    return 0;
}
