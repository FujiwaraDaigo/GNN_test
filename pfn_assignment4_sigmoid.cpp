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
    double LossFunction(int);//引数は(y)
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
public:
    double LossFunction(MatrixXd,int,MatrixXd,MatrixXd,double);//引数は(G,y,W,A,b)
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


class mSGD{
private:
    G_y_sets datasets_;
    MatrixXd W_;
    MatrixXd A_;
    double b_;
public:
    mSGD(G_y_sets,MatrixXd,MatrixXd,double);
    void Learn();
    MatrixXd getW_();
    MatrixXd getA_();
    double getb_();    
};

class Validation{
private:
    G_y_sets datasets_;
    MatrixXd W_;
    MatrixXd A_;
    double b_;
    double Loss_[2000];
    int accuracy_[2000];
public:
    Validation(G_y_sets datasets,MatrixXd W,MatrixXd A,double b);
    void validate();
    double getLossi(int);
    int getaccuracyi(int);
};



int main(){
    
    //グラフ，初期値設定

    int D=8;//特徴量ベクトルの次元
    int dataN=1600;//学習に使ったデータ

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

    G_y_sets datasets(dataN);//データ読み込み

    mSGD sgd(datasets,W,A,b);
    
    //学習
    sgd.Learn();

   
    cout << "W=" << endl;
    cout << sgd.getW_() << endl;

    cout << "A=" << endl;
    cout << sgd.getA_() << endl;

    cout << "b=" << endl;
    cout << sgd.getb_() << endl;

    //学習した値を取得
    MatrixXd Wopt=sgd.getW_();
    MatrixXd Aopt=sgd.getA_();
    double bopt=sgd.getb_();
/*
    MatrixXd Wopt(D,D);
     Wopt << 
    -0.095142 , -0.0911911 ,  -0.510538  ,  0.358249 , -0.0247866 ,  0.0684282 ,   0.460902  ,  0.394259,
    -0.148564 ,  -0.243959 ,   0.351028  , -0.466575 ,  -0.221873  ,  0.182899  , -0.522276 ,  -0.240902,
    0.424064 , -0.0336003 , -0.0216946 ,   0.241861 ,   0.343524  , -0.845173  ,  0.432257 ,  0.0450557,
    0.352425 ,   0.294288  , -0.473357 ,  -0.322587 ,  0.0282434 , -0.0305155 ,  0.0946065 , -0.0776774,
    -0.0766764  ,-0.0755851  , -0.239597 ,   0.334428 , 0.00339307 ,  0.0158877 ,    1.12959 ,   0.690849,
    -0.00738401  ,  0.286051  , -0.227891 , -0.0744365  ,  0.217405 ,   0.260346 ,  0.0947723 ,  0.0685398,
    -0.0546961  ,-0.0628451  , 0.0932854 , -0.0618302  , 0.0140735 ,  -0.579393 , -0.0617633  ,  -1.20567,
    0.091511  ,  0.713332 ,  -0.210079 ,  -0.157601 , -0.0554699 ,   0.152734 ,  -0.201572  , -0.229429;

    MatrixXd Aopt(1,D);
    Aopt <<
    0.407469  , 0.302299 , -0.106878,  -0.599544  , 0.187306 ,  0.108472 ,-0.0162092  ,-0.168684;

    double bopt=-0.0200957;
*/
    G_y_sets vdatasets(2000);//再度データ読みこみ

    Validation valid(vdatasets,Wopt,Aopt,bopt);
    valid.validate();

    double meanLoss_l=0;//学習用データの平均損失
    double accuracy_l=0;//学習用データの精度

    for(int i=0;i<dataN;i++){
        meanLoss_l+=valid.getLossi(i);
        accuracy_l+=valid.getaccuracyi(i);
    }

    meanLoss_l=meanLoss_l/dataN;
    accuracy_l=accuracy_l/dataN;

    double meanLoss_t=0;
    double accuracy_t=0;

    for(int i=dataN;i<2000;i++){
        meanLoss_t+=valid.getLossi(i);
        accuracy_t+=valid.getaccuracyi(i);
    }
    
    meanLoss_t=meanLoss_t/(2000-dataN);
    accuracy_t=accuracy_t/(2000-dataN);

    cout << "meanLoss_l=" << meanLoss_l << endl;
    cout << "accuracy_l=" << accuracy_l << endl;
    cout << "meanLoss_t=" << meanLoss_t << endl;
    cout << "accuracy_t=" << accuracy_t << endl;
     
    return 0;
}










Graph::Graph(MatrixXd G,MatrixXd W):dim_(W.rows()),v_num_(G.rows()){
    if((G.transpose() != G) || (W.rows() != W.cols())){
        std::cout << "error!:Input Matrix Dimension is Maldefined! " << std::endl;
        std::exit(1);
    }//Matrixのサイズ確認
    G_=G;
    W_=W; 
    X_= MatrixXd::Ones(dim_,v_num_);//特徴量ベクトルの初期値は一番上の行が１で他は０
}

MatrixXd Graph::concentration(int T){

    MatrixXd Z=MatrixXd::Zero(dim_,v_num_);

    for(int i=0;i<T;i++){
        MatrixXd A=X_*G_;//集約1
        MatrixXd T=W_*A;
        for(int i=0;i<T.rows();i++){
            for(int j=0;j<T.cols();j++){
                X_(i,j)=1/(1+exp(-T(i,j)));//シグモイド
            }
        }

   
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

double Classifier::LossFunction(int y){
    double p = classify();

    double r;

    if(p==-1 || p==1 || p==0){
        double s=s_;
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
    double s=classifierx.getValues_();
    double r;

    if((p==-1 || p==1) || p==0){
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



mSGD::mSGD(G_y_sets datasets,MatrixXd W,MatrixXd A,double b):datasets_(datasets.getdataN()){
    datasets_=datasets;
    W_=W;
    A_=A;
    b_=b;
}

void mSGD::Learn(){
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    int D=W_.cols();
    int epoch=20;//エポック数
    double alpha=0.0001;//学習率
    int B=512;//バッチサイズ
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
    double moment=0.9;//モーメント

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

        double L;
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
                L=PU.LossFunction(G,y,W_,A_,b_);
            }
            cout << L << " " << i << " " << k << endl;
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
        
        //余りデータがあるとき//処理しないほうがいい？
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
                L=PU.LossFunction(G,y,W_,A_,b_);
            }
            cout << L << " " << batN << " " << k << endl;
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

MatrixXd mSGD::getW_(){
    return W_;
}
    
MatrixXd mSGD::getA_(){
    return A_;
}
    
double mSGD::getb_(){
    return b_;
}

Validation::Validation(G_y_sets datasets,MatrixXd W,MatrixXd A,double b):datasets_(datasets.getdataN()){
    datasets_=datasets;
    W_=W;
    A_=A;
    b_=b;
    for(int i=0;i<2000;i++){
        Loss_[i]=0;
        accuracy_[i]=0;
    }
}

void Validation::validate(){
    for(int i=0;i<2000;i++){
        MatrixXd G=datasets_.getGi(i);
        int y=datasets_.getyi(i);
        Classifier classifier(G,W_,A_,b_);
        Loss_[i]=classifier.LossFunction(y);
    double p=classifier.classify();
    int y_c =(p>0.5);

        accuracy_[i]=(y == y_c);//pが正解に近ければaccuracy=1
        cout << y <<" "<<y_c<<" "<<p<<endl;
    }   
}

double Validation::getLossi(int i){
    return Loss_[i]; 
}

int Validation::getaccuracyi(int i){
    return accuracy_[i];
}