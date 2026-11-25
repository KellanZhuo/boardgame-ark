# app.py - 完整版Flask应用（包含收藏功能）
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, render_template_string, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import logging
import json
import os
import uuid
from datetime import datetime
import sys

# 先检查是否存在增强推荐系统文件
try:
    from enhanced_recommendation import EnhancedRecommendationSystem, BGGImageService

    HAS_ENHANCED_SYSTEM = True
except ImportError as e:
    print(f"警告: 无法导入增强推荐系统: {e}")
    print("将使用简化版本运行")
    HAS_ENHANCED_SYSTEM = False

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', 'boardgame-ark-secret-key-2026')

# 数据库配置（兼容容器无写权限目录）
default_db_path = os.path.join(os.environ.get('DATA_DIR', app.instance_path), 'boardgame_recommendations.db')
os.makedirs(os.path.dirname(default_db_path), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{default_db_path}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
db = SQLAlchemy(app)

# 初始化LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录以访问此页面'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 数据库模型
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # 关系：一个用户可以有多个保存的推荐
    saved_recommendations = db.relationship('SavedRecommendation', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
class SavedRecommendation(db.Model):
    __tablename__ = 'saved_recommendations'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    preferences = db.Column(db.Text, nullable=False)
    recommendations = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    title = db.Column(db.String(200))

    def to_dict(self):
        return {
            'id': self.id,
            'preferences': json.loads(self.preferences),
            'recommendations': json.loads(self.recommendations),
            'created_at': self.created_at.isoformat(),
            'title': self.title
        }


# 全局变量
recommender = None
image_service = None
data_loaded = False


def load_data():
    """加载推荐系统数据"""
    global recommender, image_service, data_loaded

    if not HAS_ENHANCED_SYSTEM:
        logger.warning("增强推荐系统不可用，跳过数据加载")
        data_loaded = False
        return False

    try:
        logger.info("开始加载推荐系统...")
        recommender = EnhancedRecommendationSystem()
        image_service = BGGImageService()

        # 查找数据文件 - 多个可能的路径
        possible_paths = [
            'data/BGG_Data.csv',
            'BGG_Data.csv',
            '../data/BGG_Data.csv',
            './data/BGG_Data.csv'
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break

        if not data_path:
            logger.error("未找到BGG数据文件，请确保BGG_Data.csv存在")
            data_loaded = False
            return False

        # 优先查找轻量预计算协同过滤索引，其次回退原始用户评分 CSV
        user_ratings_paths = [
            'data/cf_topk.json.gz',
            'data/cf_topk.json',
            'data/user_ratings.csv',
            'user_ratings.csv',
            '../data/user_ratings.csv',
            './data/user_ratings.csv'
        ]

        user_ratings_path = None
        for path in user_ratings_paths:
            if os.path.exists(path):
                user_ratings_path = path
                break

        if user_ratings_path:
            logger.info(f"找到用户评分数据文件: {user_ratings_path}")
        else:
            logger.warning("未找到用户评分数据文件，将仅使用内容过滤推荐")

        logger.info(f"使用数据文件: {data_path}")
        recommender.load_and_preprocess_data(data_path, user_ratings_path)
        data_loaded = True
        logger.info("推荐系统加载完成")
        return True

    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        data_loaded = False
        return False


def init_db():
    """初始化数据库"""
    with app.app_context():
        db.create_all()
        logger.info("数据库初始化完成")


# 启动时加载数据
with app.app_context():
    init_db()
load_data()


# ===== 用户认证路由 =====

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if current_user.is_authenticated:
        return redirect(url_for('wizard'))
    
    if request.method == 'POST':
        data = request.json if request.is_json else request.form
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            if request.is_json:
                return jsonify({'error': '用户名和密码不能为空'}), 400
            flash('用户名和密码不能为空')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            if request.is_json:
                return jsonify({'success': True, 'redirect': next_page or url_for('wizard')})
            return redirect(next_page) if next_page else redirect(url_for('wizard'))
        else:
            if request.is_json:
                return jsonify({'error': '用户名或密码错误'}), 401
            flash('用户名或密码错误')
            return render_template('login.html')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册"""
    if current_user.is_authenticated:
        return redirect(url_for('wizard'))
    
    if request.method == 'POST':
        data = request.json if request.is_json else request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            if request.is_json:
                return jsonify({'error': '所有字段都是必填的'}), 400
            flash('所有字段都是必填的')
            return render_template('register.html')
        
        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            if request.is_json:
                return jsonify({'error': '用户名已存在'}), 400
            flash('用户名已存在')
            return render_template('register.html')
        
        # 检查邮箱是否已存在
        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({'error': '邮箱已被注册'}), 400
            flash('邮箱已被注册')
            return render_template('register.html')
        
        # 创建新用户
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            login_user(user)
            
            if request.is_json:
                return jsonify({'success': True, 'redirect': url_for('wizard')})
            flash('注册成功！欢迎来到桌游方舟！')
            return redirect(url_for('wizard'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"注册失败: {e}")
            if request.is_json:
                return jsonify({'error': '注册失败，请重试'}), 500
            flash('注册失败，请重试')
            return render_template('register.html')
    
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    """用户登出"""
    logout_user()
    flash('nav.logout_success')
    return redirect(url_for('login'))


@app.route('/profile')
@login_required
def profile():
    """用户个人资料"""
    return render_template('profile.html', user=current_user)


@app.route('/api/user/info')
@login_required
def get_user_info():
    """获取当前用户信息API"""
    return jsonify(current_user.to_dict())


@app.route('/')
def index():
    """主页 - 重定向到向导"""
    return redirect(url_for('wizard'))


@app.route('/wizard')
def wizard():
    """推荐向导页面"""
    if not data_loaded:
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head><title>系统初始化中</title></head>
        <body>
            <h1>系统初始化中...</h1>
            <p>推荐系统正在加载数据，请稍候...</p>
            <p>如果长时间未加载，请检查：</p>
            <ul>
                <li>BGG_Data.csv 文件是否存在</li>
                <li>enhanced_recommendation.py 文件是否存在</li>
                <li>依赖包是否正确安装</li>
            </ul>
            <script>setTimeout(function(){ location.reload(); }, 10000);</script>
        </body>
        </html>
        """)

    try:
        return render_template('wizard.html')
    except Exception as e:
        logger.error(f"渲染wizard.html失败: {e}")
        return f"模板文件错误: {e}<br>请确保templates/wizard.html文件存在且格式正确"


@app.route('/recommendations')
def recommendations():
    """推荐结果页面"""
    if not data_loaded:
        return redirect(url_for('wizard'))

    try:
        return render_template('recommendations.html')
    except Exception as e:
        logger.error(f"渲染recommendations.html失败: {e}")
        return f"模板文件错误: {e}<br>请确保templates/recommendations.html文件存在且格式正确"


@app.route('/saved')
def saved_recommendations():
    """收藏页面"""
    try:
        return render_template('saved.html')
    except Exception as e:
        logger.error(f"渲染saved.html失败: {e}")
        return f"模板文件错误: {e}<br>请确保templates/saved.html文件存在且格式正确"


@app.route('/api/classic-games')
def get_classic_games():
    """获取经典游戏列表API - 精选各机制类型的代表作"""
    if not data_loaded:
        return jsonify({'error': '系统未初始化', 'games': []}), 500

    try:
        # 定义各类游戏的代表作ID（确保多样性）
        classic_game_ids = [
            174430,  # Gloomhaven - 角色扮演/战役
            167791,  # Terraforming Mars - 引擎构建
            161936,  # Pandemic Legacy: Season 1 - 合作/传承
            169786,  # Scythe - 区域控制
            120677,  # Terra Mystica - 策略建设
            31260,  # Agricola - 工人放置
            68448,  # Carcassonne - 板块拼放
            178900,  # Codenames - 聚会/词汇
            6249,  # 7 Wonders - 卡牌轮抽
            521,  # Ticket to Ride - 家庭/路线建设
            13,  # Settlers of Catan - 贸易/谈判
            36218,  # Dominion - 卡组构建
            224517,  # Azul - 抽象/图案
            266192,  # Wingspan - 引擎构建/自然主题
            170216,  # Splendor - 引擎构建/宝石
            220308,  # Brass: Birmingham - 经济策略
            295947,  # Dune: Imperium - 工人放置/卡组构建
            316554,  # Everdell - 工人放置/手牌管理
            295770,  # Root - 不对称游戏
            30549,  # Pandemic - 合作游戏
            148228,  # Love Letter - 推理/快速
            39856,  # Dixit - 创意/聚会
            230802,  # Kingdomino - 家庭/板块
            182028,  # Through the Ages - 文明建设
        ]

        games_list = []

        # 先尝试获取预定义的经典游戏
        if recommender and recommender.df is not None:
            for game_id in classic_game_ids:
                try:
                    game = recommender.df.loc[game_id]
                    if pd.notna(game['Name']):
                        game_data = {
                            'id': int(game_id),
                            'name': str(game['Name']),
                            'rating': float(game.get('Rating Average', 0)),
                            'users_rated': int(game.get('Users Rated', 0)),
                            'min_players': int(game.get('Min Players', 1)) if pd.notna(game.get('Min Players')) else 1,
                            'max_players': int(game.get('Max Players', 4)) if pd.notna(game.get('Max Players')) else 4,
                            'players_text': f"{int(game.get('Min Players', 1)) if pd.notna(game.get('Min Players')) else 1}-{int(game.get('Max Players', 4)) if pd.notna(game.get('Max Players')) else 4}人"
                        }
                        games_list.append(game_data)
                except:
                    continue

        # 如果预定义游戏不足24个，补充高评分游戏
        if len(games_list) < 24 and recommender and recommender.df is not None:
            existing_ids = [g['id'] for g in games_list]
            additional_games = recommender.df[
                (~recommender.df.index.isin(existing_ids)) &
                (recommender.df['Rating Average'] >= 7.5) &
                (recommender.df['Users Rated'] >= 1000)
                ].sort_values(['Rating Average', 'Users Rated'], ascending=[False, False])

            for idx, game in additional_games.iterrows():
                if len(games_list) >= 24:
                    break

                game_data = {
                    'id': int(idx),
                    'name': str(game['Name']),
                    'rating': float(game.get('Rating Average', 0)),
                    'users_rated': int(game.get('Users Rated', 0)),
                    'min_players': int(game.get('Min Players', 1)) if pd.notna(game.get('Min Players')) else 1,
                    'max_players': int(game.get('Max Players', 4)) if pd.notna(game.get('Max Players')) else 4,
                    'players_text': f"{int(game.get('Min Players', 1)) if pd.notna(game.get('Min Players')) else 1}-{int(game.get('Max Players', 4)) if pd.notna(game.get('Max Players')) else 4}人"
                }
                games_list.append(game_data)

        # 按评分排序
        games_list.sort(key=lambda x: x['rating'], reverse=True)

        return jsonify({'games': games_list[:24]})

    except Exception as e:
        logger.error(f"获取经典游戏时出错: {e}")
        # 返回示例数据以防止前端崩溃
        sample_games = [
            {'id': 174430, 'name': 'Gloomhaven', 'rating': 8.79, 'users_rated': 42055, 'players_text': '1-4人'},
            {'id': 30549, 'name': 'Pandemic', 'rating': 7.61, 'users_rated': 102214, 'players_text': '2-4人'},
            {'id': 167791, 'name': 'Terraforming Mars', 'rating': 8.43, 'users_rated': 64864, 'players_text': '1-5人'},
            {'id': 6249, 'name': '7 Wonders', 'rating': 7.75, 'users_rated': 80000, 'players_text': '2-7人'},
            {'id': 224517, 'name': 'Azul', 'rating': 7.83, 'users_rated': 50000, 'players_text': '2-4人'},
            {'id': 266192, 'name': 'Wingspan', 'rating': 8.11, 'users_rated': 45000, 'players_text': '1-5人'},
            {'id': 169786, 'name': 'Scythe', 'rating': 8.24, 'users_rated': 60000, 'players_text': '1-5人'},
            {'id': 120677, 'name': 'Terra Mystica', 'rating': 8.14, 'users_rated': 40000, 'players_text': '2-5人'},
            {'id': 220308, 'name': 'Brass: Birmingham', 'rating': 8.66, 'users_rated': 30000, 'players_text': '2-4人'},
            {'id': 182028, 'name': 'Through the Ages', 'rating': 8.48, 'users_rated': 25000, 'players_text': '2-4人'},
            {'id': 316554, 'name': 'Everdell', 'rating': 8.22, 'users_rated': 35000, 'players_text': '1-4人'},
            {'id': 161936, 'name': 'Pandemic Legacy S1', 'rating': 8.61, 'users_rated': 50000, 'players_text': '2-4人'},
            {'id': 178900, 'name': 'Codenames', 'rating': 7.62, 'users_rated': 80000, 'players_text': '2-8人'},
            {'id': 68448, 'name': 'Carcassonne', 'rating': 7.42, 'users_rated': 90000, 'players_text': '2-5人'},
            {'id': 31260, 'name': 'Agricola', 'rating': 7.94, 'users_rated': 60000, 'players_text': '1-5人'},
            {'id': 295770, 'name': 'Root', 'rating': 8.05, 'users_rated': 40000, 'players_text': '2-4人'},
            {'id': 167355, 'name': 'Viticulture', 'rating': 8.11, 'users_rated': 35000, 'players_text': '1-6人'},
            {'id': 295947, 'name': 'Dune: Imperium', 'rating': 8.35, 'users_rated': 25000, 'players_text': '1-4人'},
            {'id': 102794, 'name': 'Power Grid', 'rating': 7.82, 'users_rated': 50000, 'players_text': '2-6人'},
            {'id': 13, 'name': 'Settlers of Catan', 'rating': 7.13, 'users_rated': 100000, 'players_text': '3-4人'},
            {'id': 521, 'name': 'Ticket to Ride', 'rating': 7.42, 'users_rated': 85000, 'players_text': '2-5人'},
            {'id': 192135, 'name': 'Great Western Trail', 'rating': 8.27, 'users_rated': 30000,
             'players_text': '2-4人'},
            {'id': 36218, 'name': 'Dominion', 'rating': 7.59, 'users_rated': 70000, 'players_text': '2-4人'},
            {'id': 170216, 'name': 'Splendor', 'rating': 7.41, 'users_rated': 65000, 'players_text': '2-4人'}
        ]
        return jsonify({'games': sample_games})


@app.route('/api/recommendations', methods=['POST'])
def generate_recommendations():
    """生成推荐API"""
    if not data_loaded:
        return jsonify({'error': '系统未初始化'}), 500

    try:
        preferences = request.json
        logger.info(f"收到推荐请求: {preferences}")

        # 验证请求数据
        if not preferences:
            return jsonify({'error': '无效的偏好数据'}), 400

        # 生成主推荐
        main_recommendations = recommender.get_enhanced_recommendations(preferences, N=12) if recommender else []

        # 获取高评分游戏
        top_rated = recommender.get_top_rated_games(N=4) if recommender else []

        # 获取最新游戏
        newest_games = recommender.get_newest_games(N=4) if recommender else []

        # 获取更多匹配游戏
        more_matches = recommender.get_enhanced_recommendations(preferences, N=20)[12:16] if recommender else []

        # 存储到session中
        session['last_recommendations'] = {
            'main': main_recommendations,
            'top_rated': top_rated,
            'newest': newest_games,
            'more_matches': more_matches,
            'preferences': preferences,
            'timestamp': datetime.now().isoformat()
        }

        response_data = {
            'main_recommendations': main_recommendations,
            'top_rated_games': top_rated,
            'newest_games': newest_games,
            'more_matches': more_matches,
            'total_count': len(main_recommendations),
            'status': 'success'
        }

        logger.info(f"成功生成推荐结果: 主推荐{len(main_recommendations)}个")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"生成推荐时出错: {e}")
        # 返回示例数据
        sample_data = {
            'main_recommendations': [
                {'id': 1, 'name': 'Gloomhaven', 'rating': 8.79, 'year': 2017},
                {'id': 2, 'name': 'Pandemic', 'rating': 7.61, 'year': 2008},
            ],
            'top_rated_games': [
                {'id': 1, 'name': 'Gloomhaven', 'rating': 8.79, 'year': 2017},
            ],
            'newest_games': [
                {'id': 3, 'name': 'Wingspan', 'rating': 8.11, 'year': 2019},
            ],
            'more_matches': [
                {'id': 4, 'name': 'Scythe', 'rating': 8.24, 'year': 2016},
            ],
            'total_count': 2,
            'status': 'success'
        }
        return jsonify(sample_data)


@app.route('/api/game-image/<int:game_id>')
def get_game_image(game_id):
    """获取单个游戏图片URL API"""
    if not data_loaded or not image_service:
        return jsonify({'image_url': None}), 404

    try:
        image_url = image_service.get_game_image_url(game_id)
        if image_url:
            return jsonify({'image_url': image_url, 'game_id': game_id})
        else:
            return jsonify({'image_url': None, 'game_id': game_id}), 404

    except Exception as e:
        logger.error(f"获取游戏图片时出错: {e}")
        return jsonify({'error': str(e), 'game_id': game_id}), 500


@app.route('/api/games/images', methods=['POST'])
def get_multiple_game_images():
    """批量获取游戏图片URL API"""
    if not data_loaded or not image_service:
        return jsonify({'images': {}})

    try:
        data = request.get_json()
        game_ids = data.get('game_ids', [])

        if not game_ids:
            return jsonify({'images': {}})

        # 限制批量请求数量
        game_ids = game_ids[:20]  # 最多20个

        images = image_service.get_multiple_image_urls(game_ids)

        return jsonify({
            'images': images,
            'count': len(images),
            'requested': len(game_ids)
        })

    except Exception as e:
        logger.error(f"批量获取游戏图片时出错: {e}")
        return jsonify({'error': str(e), 'images': {}})


@app.route('/api/recommendations/last')
def get_last_recommendations():
    """获取上次的推荐结果"""
    try:
        last_recommendations = session.get('last_recommendations')

        if not last_recommendations:
            return jsonify({'error': '没有找到推荐数据'}), 404

        # 检查推荐数据是否过期（比如超过1小时）
        from datetime import datetime, timedelta
        timestamp_str = last_recommendations.get('timestamp')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if datetime.now() - timestamp > timedelta(hours=1):
                return jsonify({'error': '推荐数据已过期'}), 404

        response_data = {
            'main_recommendations': last_recommendations.get('main', []),
            'top_rated_games': last_recommendations.get('top_rated', []),
            'newest_games': last_recommendations.get('newest', []),
            'more_matches': last_recommendations.get('more_matches', []),
            'preferences': last_recommendations.get('preferences', {}),
            'timestamp': timestamp_str
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"获取上次推荐时出错: {e}")
        return jsonify({'error': '获取推荐数据失败'}), 500


@app.route('/api/games/more')
def get_more_games():
    """获取更多游戏API - 支持不同类型的游戏列表"""
    if not data_loaded:
        return jsonify({'error': '系统未初始化', 'games': []}), 500

    game_type = request.args.get('type', 'rating')
    limit = min(int(request.args.get('limit', 20)), 50)  # 最多返回50个

    try:
        games_list = []

        if game_type == 'rating':
            # 获取更多高评分游戏
            if recommender and recommender.df is not None:
                high_rated = recommender.df[
                    (recommender.df['Rating Average'] >= 7.0) &
                    (recommender.df['Users Rated'] >= 1000)
                    ].sort_values(['Rating Average', 'Users Rated'], ascending=[False, False])

            for idx, game in high_rated.head(limit).iterrows():
                games_list.append({
                    'id': int(idx),
                    'name': str(game['Name']),
                    'rating': float(game.get('Rating Average', 0)),
                    'year': int(game.get('Year Published', 0)) if pd.notna(game.get('Year Published')) else 0,
                    'users_rated': int(game.get('Users Rated', 0)),
                    'min_players': int(game.get('Min Players', 1)) if pd.notna(game.get('Min Players')) else 1,
                    'max_players': int(game.get('Max Players', 4)) if pd.notna(game.get('Max Players')) else 4,
                    'complexity': float(game.get('Complexity', 0)) if pd.notna(
                        game.get('Complexity')) else 0
                })

        elif game_type == 'newest':
            # 获取更多新游戏
            if recommender and recommender.df is not None:
                current_year = 2025
                newest = recommender.df[
                    (recommender.df['Year Published'] >= current_year - 5) &
                    (recommender.df['Users Rated'] >= 100)
                    ].sort_values(['Year Published', 'Rating Average'], ascending=[False, False])

            for idx, game in newest.head(limit).iterrows():
                games_list.append({
                    'id': int(idx),
                    'name': str(game['Name']),
                    'rating': float(game.get('Rating Average', 0)),
                    'year': int(game.get('Year Published', 0)) if pd.notna(game.get('Year Published')) else 0,
                    'users_rated': int(game.get('Users Rated', 0)),
                    'min_players': int(game.get('Min Players', 1)) if pd.notna(game.get('Min Players')) else 1,
                    'max_players': int(game.get('Max Players', 4)) if pd.notna(game.get('Max Players')) else 4,
                    'play_time': int(game.get('Play Time', 0)) if pd.notna(game.get('Play Time')) else 0
                })

        elif game_type == 'matches':
            # 获取更多匹配游戏（从会话中获取）
            last_recommendations = session.get('last_recommendations', {})

            if last_recommendations and 'preferences' in last_recommendations:
                # 生成更多推荐
                if recommender:
                    preferences = last_recommendations['preferences']
                    more_recommendations = recommender.get_enhanced_recommendations(preferences, N=limit + 12)

                # 跳过前12个（已经在主推荐中显示）
                for rec in more_recommendations[12:]:
                    games_list.append(rec)
            else:
                # 如果没有会话数据，返回高评分游戏作为替代
                return get_more_games_fallback('rating', limit)

        logger.info(f"返回 {len(games_list)} 个 {game_type} 类型的游戏")
        return jsonify({'games': games_list, 'type': game_type, 'count': len(games_list)})

    except Exception as e:
        logger.error(f"获取更多游戏时出错: {e}")
        return jsonify({'error': str(e), 'games': []}), 500


def get_more_games_fallback(game_type, limit):
    """获取更多游戏的备用方案"""
    sample_games = []

    if game_type == 'rating':
        base_games = [
            {'id': 174430, 'name': 'Gloomhaven', 'rating': 8.79, 'year': 2017},
            {'id': 220308, 'name': 'Brass: Birmingham', 'rating': 8.66, 'year': 2018},
            {'id': 161936, 'name': 'Pandemic Legacy: Season 1', 'rating': 8.61, 'year': 2015},
            {'id': 167791, 'name': 'Terraforming Mars', 'rating': 8.43, 'year': 2016},
            {'id': 182028, 'name': 'Through the Ages', 'rating': 8.48, 'year': 2014}
        ]
    elif game_type == 'newest':
        base_games = [
            {'id': 295947, 'name': 'Dune: Imperium', 'rating': 8.35, 'year': 2020},
            {'id': 266192, 'name': 'Wingspan', 'rating': 8.11, 'year': 2019},
            {'id': 316554, 'name': 'Everdell', 'rating': 8.22, 'year': 2021},
            {'id': 295770, 'name': 'Root', 'rating': 8.05, 'year': 2018}
        ]
    else:
        base_games = [
            {'id': 167791, 'name': 'Terraforming Mars', 'rating': 8.43, 'year': 2016},
            {'id': 120677, 'name': 'Terra Mystica', 'rating': 8.14, 'year': 2012},
            {'id': 31260, 'name': 'Agricola', 'rating': 7.94, 'year': 2007}
        ]

    # 生成更多示例数据
    for i in range(limit):
        base = base_games[i % len(base_games)]
        game = base.copy()
        if i >= len(base_games):
            game['name'] = f"{game['name']} (示例 {i + 1})"
            game['id'] = game['id'] + i * 1000
        sample_games.append(game)

    return jsonify({'games': sample_games, 'type': game_type, 'count': len(sample_games)})


# ===== 收藏功能相关路由 =====

@app.route('/api/recommendations/save', methods=['POST'])
@login_required
def save_recommendation():
    """保存推荐结果到数据库"""
    try:
        data = request.json

        # 使用当前登录用户的ID
        user_id = current_user.id

        # 创建保存记录
        saved_rec = SavedRecommendation()
        saved_rec.user_id = user_id
        saved_rec.preferences = json.dumps(data.get('preferences', {}))
        saved_rec.recommendations = json.dumps(data.get('recommendations', {}))
        saved_rec.title = data.get('title', f"推荐_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        db.session.add(saved_rec)
        db.session.commit()

        return jsonify({
            'success': True,
            'id': saved_rec.id,
            'message': '推荐已保存'
        })

    except Exception as e:
        logger.error(f"保存推荐失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations/saved')
@login_required
def get_saved_recommendations():
    """获取用户保存的推荐列表"""
    try:
        user_id = current_user.id

        saved_recs = SavedRecommendation.query.filter_by(
            user_id=user_id
        ).order_by(SavedRecommendation.created_at.desc()).limit(20).all()

        return jsonify({
            'recommendations': [rec.to_dict() for rec in saved_recs]
        })

    except Exception as e:
        logger.error(f"获取保存的推荐失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations/saved/<int:rec_id>')
@login_required
def get_saved_recommendation_detail(rec_id):
    """获取特定保存的推荐详情"""
    try:
        user_id = current_user.id

        saved_rec = SavedRecommendation.query.filter_by(
            id=rec_id,
            user_id=user_id
        ).first()

        if not saved_rec:
            return jsonify({'error': '未找到记录'}), 404

        return jsonify(saved_rec.to_dict())

    except Exception as e:
        logger.error(f"获取推荐详情失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations/saved/<int:rec_id>', methods=['DELETE'])
@login_required
def delete_saved_recommendation(rec_id):
    """删除保存的推荐"""
    try:
        user_id = current_user.id

        saved_rec = SavedRecommendation.query.filter_by(
            id=rec_id,
            user_id=user_id
        ).first()

        if not saved_rec:
            return jsonify({'error': '未找到记录'}), 404

        db.session.delete(saved_rec)
        db.session.commit()

        return jsonify({'success': True, 'message': '删除成功'})

    except Exception as e:
        logger.error(f"删除推荐失败: {e}")
        return jsonify({'error': str(e)}), 500


# ===== 通用路由 =====

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy' if data_loaded else 'initializing',
        'data_loaded': data_loaded,
        'enhanced_system': HAS_ENHANCED_SYSTEM,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '页面未找到'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"内部服务器错误: {error}")
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("桌游方舟推荐系统启动中...")
    print(f"增强推荐系统: {'可用' if HAS_ENHANCED_SYSTEM else '不可用'}")
    print(f"数据加载状态: {'成功' if data_loaded else '失败'}")
    print("=" * 50)

    if data_loaded:
        print("✅ 系统启动成功！")
        print("📱 访问地址: http://localhost:8080")
    else:
        print("⚠️  系统启动但数据未加载")
        print("🔧 请检查：")
        print("   1. BGG_Data.csv 文件是否存在")
        print("   2. enhanced_recommendation.py 文件是否存在")
        print("   3. 依赖包是否正确安装")

    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)