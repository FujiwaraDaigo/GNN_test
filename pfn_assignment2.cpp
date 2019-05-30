#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cstdlib>
#include <random>

using namespace std; 
using namespace Eigen;

//初期値やパラメータの入力はmain内で行う

//グラフクラスの定義
class Graph{
private:
    int dim_;//特徴量ベクトルの次元
    int v_num_;//頂点数
    MatrixXd G_;//行列形式でのグラフ表現，ij成分はノードiとノードjが繋がっているとき1，それ以外と対角成分は0
    MatrixXd W_;//パラメータ行列
    MatrixXd X_;//特徴量ベクトルを並べた行列
    VectorXd hg_;//グラフGに対する特徴量ベクトル
public:
    Graph(MatrixXd G,MatrixXd W);//コンストラクタ,引数は(G,W)
    MatrixXd concentration(int);//集約,引数は集約回数 
};

//分類器クラスの定義
class Classifier{
private:
    Graph graph_;//分類器のもつグラフ
    MatrixXd G_;//グラフ
    MatrixXd W_;//重みパラメータ
    MatrixXd A_;//重みパラメータ
    double b_;//バイアスパラメータ
    double s_;//分類結果(計算前) 
    double p_;//分類結果
    int T_;//集約回数
public:
    Classifier(MatrixXd G,MatrixXd W,MatrixXd A, double b);//コンストラクタ，引数は(G,W,A,b)
    double classify();//分類出力関数
    double getValues_();//getter
    MatrixXd getMatrixG_();//各行列を出力するgetter
    MatrixXd getMatrixW_();
    MatrixXd getMatrixA_();
    double getValueb_();
};

class ParameterUpdate{
private:
    int dim_;//特徴量ベクトルの次元
    Classifier classifier_; //分類器
    MatrixXd G_;//グラフ 
    int y_;//ラベル
    MatrixXd dW_;//各更新量
    MatrixXd dA_;
    double db_;
    double LossOld_;
    double LossNew_;
    double LossFunction(MatrixXd,int,MatrixXd,MatrixXd,double);//引数は(G,y,W,A,b)
    void differentiation(MatrixXd G,double y,MatrixXd W,MatrixXd A,double b); //微分操作
public:
    ParameterUpdate(MatrixXd,double,MatrixXd,MatrixXd,double);//コンストラクタ，引数は(G,y,W,A,b)
    Classifier getClassifier();
    double getLossOld();
    void update();
};





int main(){
    //グラフ，初期値設定

    int V=8;//グラフの頂点数
    //適当なグラフ生成
    MatrixXd G(V,V);//行列形式でのグラフ表現，ij成分はノードiとノードjが繋がっているとき1，それ以外と対角成分は0
    
    for(int i=0;i<V;i++){
        for(int j=0;j<V;j++){
            G(i,j)= (std::rand()%2);
            G(j,i)=G(i,j);
        }
    }

    cout << G << endl;
 

    int y=0;//適当なラベル

    int D=4;//特徴量ベクトルの次元

    //パラメータ行列のランダム初期化
    double mean=0.0;
    double stddev=0.4;
    
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> n_dist(mean,stddev);

    MatrixXd W0(D,D);//パラメータ行列 
    for(int i=0;i<D;i++){
        for(int j=0;j<D;j++){
            W0(i,j)= n_dist(engine);
        }
    }

    MatrixXd A0(1,D);//パラメータ行列
    for(int j=0;j<D;j++){
        A0(0,j)= n_dist(engine);
    }

    double b0=0;//パラメータ初期化

    ParameterUpdate PU(G,y,W0,A0,b0);
    
    int count = 0;
    while(1){
        PU.update();//パラメータを学習
        cout << PU.getLossOld() << endl;
        if((PU.getLossOld()<0.01) || (count>1000) ){
            break;
        }//損失が0.01以下になるかループが1000回を超えたら終了
        count++;
    }

    Classifier ResultClassifier = PU.getClassifier();//得られた分類器を取得

    cout << "W=" << endl;
    cout << ResultClassifier.getMatrixW_() << endl;

    cout << "A=" << endl;
    cout << ResultClassifier.getMatrixA_() << endl;

    cout << "b=" << endl;
    cout << ResultClassifier.getValueb_() << endl;
    
    return 0;
}










Graph::Graph(MatrixXd G,MatrixXd W):dim_(W.rows()),v_num_(G.rows()){
    if((G.transpose() != G) || (W.rows() != W.cols())){
        std::cout << "error!:Input Matrix Dimension is Maldefined! " << std::endl;
        std::exit(1);
    }//Matrixのサイズ確認
    G_=G;
    W_=W; 
    X_= MatrixXd::Ones(dim_,v_num_);//特徴量ベクトルの初期値は全ての要素が１とした
}

MatrixXd Graph::concentration(int T){

    MatrixXd Z=MatrixXd::Zero(dim_,v_num_);

    for(int i=0;i<T;i++){
        MatrixXd A=X_*G_;//集約1
        X_= (W_*A).cwiseMax(Z);//集約２
    }

    hg_=X_*VectorXd::Ones(v_num_);
    
    return hg_;
}

Classifier::Classifier(MatrixXd G,MatrixXd W,MatrixXd A,double b):graph_(G,W){
    if(A.cols() != W.rows()){
        std::cout << "error!Classifier:Input Matrix Dimension is Maldefined!" << std::endl;
        std::exit(1);
    }//Matrixのサイズ確認
    G_=G;
    W_=W;
    A_=A;
    b_=b;
    T_=2;//ハイパーパラメータ
}

double Classifier::classify(){
    //特徴量ベクトルhgの出力
    MatrixXd hg;
    hg = graph_.concentration(T_);//特徴量ベクトルの取得

    s_ = (A_*hg)(0,0)+b_;

    if(fabs(s_)<700){
        p_=1/(1+exp(-s_));
        return p_;
    }//オーバーフロー，アンダーフロー回避
    else {
        return -1;//オーバー・アンダーフローのフラグ，-1なら注意
    }
}

double Classifier::getValues_(){
    return s_;
}

MatrixXd Classifier::getMatrixG_(){
    return G_;
}

MatrixXd Classifier::getMatrixW_(){
    return W_;
}

MatrixXd Classifier::getMatrixA_(){
    return A_;
}

double Classifier::getValueb_(){
    return b_;
}

ParameterUpdate::ParameterUpdate(MatrixXd G,double y,MatrixXd W0,MatrixXd A0,double b0):classifier_(G,W0,A0,b0),dim_(W0.rows()){
    dW_ = MatrixXd::Zero(dim_,dim_);
    dA_ = MatrixXd::Zero(1,dim_);
    db_ = 0;
    G_ = G;
    y_ = y;
}//コンストラクタ，引数は(G,y,W,A,b)

double ParameterUpdate::LossFunction(MatrixXd G,int y,MatrixXd W,MatrixXd A,double b){
    Classifier classifierx(G,W,A,b);
    double p = classifierx.classify();

    double r;

    if(p==-1 || p==1 || p==0){
        double s=classifier_.getValues_();
        if(s>0){
            r = (1-y)*s;
        }
        else if(s<=0){
            r = -y*s;
        }
    }//オーバー・アンダーフロー回避の例外的処理

    else{
        r = -y*log(p)-(1-y)*log(1-p);
    }
    
    return r;
}

void ParameterUpdate::differentiation(MatrixXd G,double y,MatrixXd W,MatrixXd A,double b){
    double eps=0.001;
    double L=LossFunction(G,y,W,A,b);

    MatrixXd E = MatrixXd::Zero(dim_,dim_);
    for(int i=0;i<dim_;i++){
        for(int j=0;j<dim_;j++){
            E(i,j)=1;
            dW_(i,j)=(LossFunction(G,y,W+eps*E,A,b)-L)/eps;
            E(i,j)=0;
        }
    }

    MatrixXd V = MatrixXd::Zero(1,dim_);
    for(int i=0;i<dim_;i++){
        V(0,i)=1;
        dA_(0,i)=(LossFunction(G,y,W,A+eps*V,b)-L)/eps;
        V(0,i)=0;
    }

    db_ = (LossFunction(G,y,W,A,b+eps)-L)/eps;

    
    
};

void ParameterUpdate::update(){
    double alpha = 0.0001;
    Classifier classifierOld = classifier_;
    LossOld_=LossFunction(classifierOld.getMatrixG_(),y_,classifierOld.getMatrixW_(),classifierOld.getMatrixA_(),classifierOld.getValueb_());
    differentiation(classifierOld.getMatrixG_(),y_,classifierOld.getMatrixW_(),classifierOld.getMatrixA_(),classifierOld.getValueb_());
    //更新量を元に新たな分類機を生成
    Classifier classifierNew(G_,classifierOld.getMatrixW_()-alpha*dW_,classifierOld.getMatrixA_()-alpha*dA_,classifierOld.getValueb_()-alpha*db_);
    LossNew_=LossFunction(classifierNew.getMatrixG_(),y_,classifierNew.getMatrixW_(),classifierNew.getMatrixA_(),classifierNew.getValueb_());
    classifier_ = classifierNew;
}

Classifier ParameterUpdate::getClassifier(){
    return classifier_;
}

double ParameterUpdate::getLossOld(){
    return LossOld_;
}


