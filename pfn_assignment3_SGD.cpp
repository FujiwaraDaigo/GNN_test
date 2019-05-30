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
    MatrixXd G_;//グラフ 
    int y_;//ラベル
    MatrixXd dW_;//各更新量
    MatrixXd dA_;
    double db_;
    double LossOld_;
    double LossNew_;
    double LossFunction(MatrixXd,int,MatrixXd,MatrixXd,double);//引数は(G,y,W,A,b)
public:
    void differentiation(MatrixXd G,double y,MatrixXd W,MatrixXd A,double b); //微分操作
    ParameterUpdate(MatrixXd,double,MatrixXd,MatrixXd,double);//コンストラクタ，引数は(G,y,W,A,b)
    MatrixXd getMatrixdW_();
    MatrixXd getMatrixdA_();
    double getValuedb_();
};

class G_y_sets{
private:
    MatrixXd G_[2000];
    double y_[2000];
    void readData();
    int dataN_;
public:
    G_y_sets(int dataN);
    MatrixXd getGi(int);
    int getyi(int);
    int getdataN();
};

class SGD{
private:
    G_y_sets datasets_;
    MatrixXd W_;
    MatrixXd A_;
    double b_;
public:
    SGD(G_y_sets,MatrixXd,MatrixXd,double);
    void Learn();
    MatrixXd getW_();
    MatrixXd getA_();
    double getb_();    
};




int main(){
    
    //グラフ，初期値設定

    int D=8;//特徴量ベクトルの次元

    //パラメータ行列のランダム初期値作成
    double mean=0.0;
    double stddev=0.4;
    
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> n_dist(mean,stddev);//正規分布による初期化

    MatrixXd W(D,D);//パラメータ行列 
    for(int i=0;i<D;i++){
        for(int j=0;j<D;j++){
            W(i,j)= n_dist(engine);
        }
    }

    MatrixXd A(1,D);//パラメータ行列
    for(int j=0;j<D;j++){
        A(0,j)= n_dist(engine);
    }

    double b=0;//パラメータ

    int dataN=1200;

    G_y_sets datasets(dataN);//データ読み込み

    SGD sgd(datasets,W,A,b);
    
    sgd.Learn();

    cout << "W=" << endl;
    cout << sgd.getW_() << endl;

    cout << "A=" << endl;
    cout << sgd.getA_() << endl;

    cout << "b=" << endl;
    cout << sgd.getb_() << endl;
    
    return 0;
}










Graph::Graph(MatrixXd G,MatrixXd W):dim_(W.rows()),v_num_(G.rows()){
    if((G.transpose() != G) || (W.rows() != W.cols())){
        std::cout << "error!:Input Matrix Dimension is Maldefined! " << std::endl;
        std::exit(1);
    }//Matrixのサイズ確認
    G_=G;
    W_=W; 
    X_= MatrixXd::Zero(dim_,v_num_);//特徴量ベクトルの初期値は一番上の行が１で他は０
    X_.row(0)=MatrixXd::Ones(1,v_num_);
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

ParameterUpdate::ParameterUpdate(MatrixXd G,double y,MatrixXd W0,MatrixXd A0,double b0):dim_(W0.rows()){
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
        double s=classifierx.getValues_();
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







MatrixXd ParameterUpdate::getMatrixdW_(){
    return dW_;
}

MatrixXd ParameterUpdate::getMatrixdA_(){
    return dA_;
}

double ParameterUpdate::getValuedb_(){
    return db_;
}

G_y_sets::G_y_sets(int dataN):dataN_(dataN){
    readData();
}

void G_y_sets::readData(){

    char filename[40];
    FILE *fp;
    int v;//頂点数
    double data;//データ読み取り用
    int int_data;//データ読み取り用

    for(int i=0;i<dataN_;i++){
        sprintf(filename,"./datasets/train/%d_graph.txt",i);
        fp=fopen(filename,"r");
        if (fp == NULL) {
            cout << "error: can't read file." << endl;
            exit(1);
        }
        fscanf(fp,"%d",&v);//頂点数を取得
        G_[i]=MatrixXd::Zero(v,v);//G[i]の初期化
        for(int j=0;j<v;j++){
            for(int k=0;k<v;k++){
                fscanf(fp,"%lf",&data);
                G_[i](j,k)=data;
            }
        }
        fclose(fp);

        sprintf(filename,"./datasets/train/%d_label.txt",i);
        fp=fopen(filename,"r");
        if (fp == NULL) {
            cout << "error: can't read file." << endl;
            exit(1);
        }
        fscanf(fp,"%d",&int_data);
        y_[i]=int_data;
        fclose(fp);
    }
    
}

MatrixXd G_y_sets::getGi(int i){
    return G_[i];
}

int G_y_sets::getyi(int i){
    return y_[i];
}

int G_y_sets::getdataN(){
    return dataN_;
}

SGD::SGD(G_y_sets datasets,MatrixXd W,MatrixXd A,double b):datasets_(datasets.getdataN()){
    datasets_=datasets;
    W_=W;
    A_=A;
    b_=b;
}

void SGD::Learn(){
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    int D=W_.cols();
    int epoch=16;//エポック数
    double alpha=0.0001;//学習率
    int B=64;//バッチサイズ
    int dataN_=datasets_.getdataN();
    int batN=dataN_/B;//バッチの個数
    int tempa;
    int exchange;
    int a[dataN_];
    for(int i=0;i<dataN_;i++){
        a[i]=i;
    }
    MatrixXd meandW = MatrixXd::Zero(D,D);
    MatrixXd meandA = MatrixXd::Zero(1,D);
    double meandb = 0;
    MatrixXd lastdW = MatrixXd::Zero(D,D);//モーメンタム法
    MatrixXd lastdA= MatrixXd::Zero(1,D);
    double lastdb=0;
    double moment=0;//モーメント

    for(int k=0;k<epoch;k++){
        //ランダムサンプリング,G[],y[]の並び替え,並び替えて順に前からB個とる
        for(int i=0;i<dataN_;i++){

            std::uniform_int_distribution<> u_dist(i,dataN_-1); 
            exchange= u_dist(engine);//i~dataN-1から数字を一つ選ぶ
            tempa=a[i];
            a[i]=a[exchange];
            //左端とj番目を交換
            a[exchange]=tempa;
        }

        //前からB個ずつとる
        for(int i=0;i<batN;i++){
            meandW = MatrixXd::Zero(D,D);
            meandA = MatrixXd::Zero(1,D);
            meandb = 0;

            for(int j=0;j<B;j++){
                MatrixXd G=datasets_.getGi(a[i*B+j]);
                double y=datasets_.getyi(a[i*B+j]);
                ParameterUpdate PU(G,y,W_,A_,b_);
                PU.differentiation(G,y,W_,A_,b_);
                meandW += PU.getMatrixdW_();
                meandA += PU.getMatrixdA_();
                meandb += PU.getValuedb_();
            }
            meandW = meandW/B;
            meandA = meandA/B;
            meandb = meandb/B;

            lastdW = -alpha*meandW+moment*lastdW;
            lastdA = -alpha*meandA+moment*lastdA;
            lastdb = -alpha*meandb+moment*lastdb;


            W_ += lastdW;
            A_ += lastdA;
            b_ += lastdb;//パラメータの更新
        }
        
        //余りデータがあるとき//処理しない方がいい？
        if(dataN_%B){
            meandW = MatrixXd::Zero(D,D);
            meandA = MatrixXd::Zero(1,D);
            meandb = 0;   
            for(int i=batN*B;i<batN*B+dataN_%B;i++){
                MatrixXd G=datasets_.getGi(a[i]);
                double y=datasets_.getyi(a[i]);
                ParameterUpdate PU(G,y,W_,A_,b_);
                PU.differentiation(G,y,W_,A_,b_);
                meandW += PU.getMatrixdW_();
                meandA += PU.getMatrixdA_();
                meandb += PU.getValuedb_();
            }
            meandW = meandW/B;
            meandA = meandA/B;
            meandb = meandb/B;

            lastdW = -alpha*meandW+moment*lastdW;
            lastdA = -alpha*meandA+moment*lastdA;
            lastdb = -alpha*meandb+moment*lastdb;

            W_ += lastdW;
            A_ += lastdA;
            b_ += lastdb;//パラメータの更新
        }
    }

}

MatrixXd SGD::getW_(){
    return W_;
}
    
MatrixXd SGD::getA_(){
    return A_;
};
    
double SGD::getb_(){
    return b_;
}; 