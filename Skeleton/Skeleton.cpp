//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Koczka Vencel Dávid
// Neptun : RR2FJM
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

enum MaterialType {ROUGH, REFLECTIVE, PORTAL };

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct PortalMaterial : Material {
	PortalMaterial(vec3 n, vec3 kappa) : Material(PORTAL) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	vec3 center;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Paraboloid : Intersectable {

	Paraboloid(Material* _material) {
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 center(0, 0, 0);
		float a = 2.4f;
		float b = 8.2f;
		float c = 1.1f;
		float A = a*ray.dir.x*ray.dir.x + b*ray.dir.y*ray.dir.y;
		float B = 2*a*ray.start.x*ray.dir.x + 2*b*ray.start.y*ray.dir.y - c*ray.dir.z;
		float C = a*ray.start.x*ray.start.x + b*ray.start.y*ray.start.y - c*ray.start.z;
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0) return hit;
		vec3 p1 = ray.start + ray.dir * t1;
		vec3 p2 = ray.start + ray.dir * t2;
		if (length(p1) > 0.3f && length(p2) > 0.3f) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		

		if (length(p1) < 0.3f && length(p2) >= 0.3f) {
			hit.t = t1;
		}

		hit.position = ray.start + ray.dir * hit.t;
		float s = expf(a * hit.position.x * hit.position.x + b * hit.position.y * hit.position.y - c * hit.position.z);
		vec3 F(2 * a * hit.position.x, 2*b*hit.position.y, -1.0f * c);
		F = normalize(F);
		hit.normal = F;
		hit.material = material;
		return hit;
	}
};

struct Pentagon : public Intersectable {
	vec3 points[5];
	vec3 center;
	vec3 normal;

	Pentagon(vec3* _points, Material* _material) {

		for (int i = 0; i < 5; i++) {
			points[i] = _points[i];
		}
		material = _material;
		for (vec3 v : points) {
			center = center + v;
		}
		center = center / 5;
		normal = normalize(cross(points[1] - points[0], points[4] - points[0]));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float t = dot((points[0] - ray.start), normal) / dot(ray.dir, normal);
		if (t <= 0) { return hit; }
		vec3 p = ray.start + ray.dir * t;

		if (!(dot(cross(points[1] - points[0], p - points[0]), normal) > 0 &&
			dot(cross(points[2] - points[1], p - points[1]), normal) > 0 &&
			dot(cross(points[3] - points[2], p - points[2]), normal) > 0 &&
			dot(cross(points[4] - points[3], p - points[3]), normal) > 0 &&
			dot(cross(points[0] - points[4], p - points[4]), normal) > 0)) {
			return hit;
		}

		hit.t = t;
		hit.position = p;
		hit.normal = normal;
		hit.material = material;
		hit.center = center;
		return hit;
	}
};

const float epsilon = 0.0001f;

struct Portal : public Pentagon {

	Portal(vec3* _points, Material* _material) : Pentagon(_points, _material){
		for (int i = 0; i < 5; i++) {
			//points[i] = points[i] - (normalize(center - points[i]) * (0.1f / sinf(54.0f)));
			points[i] = points[i] + (normalize(center - points[i]) * 0.1f);
			points[i] = points[i] - normal * epsilon;
		}
	}
};

class DodecaHedron{
public:
	Pentagon* sides[12];
	Portal* portals[12];
	vec3 points[20] = { vec3(0, 0.618, 1.618), vec3(0, -0.618, 1.618), vec3(0, -0.618, -1.618), vec3(0, 0.618, -1.618), vec3(1.618, 0, 0.618),
						vec3(-1.618, 0, 0.618), vec3(-1.618, 0, -0.618), vec3(1.618, 0, -0.618), vec3(0.618, 1.618, 0), vec3(-0.618, 1.618, 0),
						vec3(-0.618, -1.618, 0), vec3(0.618, -1.618, 0), vec3(1,1,1), vec3(-1,1,1), vec3(-1,-1,1), 
						vec3(1,-1,1), vec3(1,-1,-1), vec3(1,1,-1), vec3(-1,1,-1), vec3(-1,-1,-1) };

	int sideNums[12][5] = { {1,2,16,5,13},
						{1,13,9,10,14},
						{1,14,6,15,2},
						{2,15,11,12,16},
						{3,4,18,8,17},
						{3,17,12,11,20},
						{3,20,7,19,4},
						{19,10,9,18,4},
						{16,12,17,8,5},
						{5,8,18,9,13},
						{14,10,19,7,6},
						{6,7,20,11,15} };

	DodecaHedron() {
		vec3 kd(0.5f, 0.3f, 0.5f), ks(0, 0, 0);
		Material* material = new RoughMaterial(kd, ks, 50);

		vec3 n(1,1,1), kappa(5,4,3);
		Material* material2 = new PortalMaterial(n, kappa);

		for (int i = 0; i < 12; i++) {
			vec3 vertexes[5];
			for (int j = 0; j < 5; j++) {
				vertexes[j] = points[sideNums[i][j]-1];
			}
			sides[i] = new Pentagon(vertexes, material);
			portals[i] = new Portal(vertexes, material2);
		}
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) - d.y * sin(dt), d.x * sin(dt) + d.y * cos(dt), d.z) + lookat;
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 position;
	vec3 Le;
	Light(vec3 _direction, vec3 _position, vec3 _Le) {
		direction = normalize(_direction);
		position = _position;
		Le = _Le;
	}
};

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(1.2f, 0, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.2f, 0.2f, 0.2f);
		vec3 lightDirection(1, 1, 1), lightPosition(0,0.5f,0.5f), Le(1, 1, 1);
		lights.push_back(new Light(lightDirection, lightPosition,  Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new RoughMaterial(kd, ks, 50);

		vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
		Material* material2 = new ReflectiveMaterial(n, kappa);

		objects.push_back(new Paraboloid(material2));
		DodecaHedron dodeka = DodecaHedron();
		for (int i = 0; i < 12; i++) {
			objects.push_back(dodeka.sides[i]);
			objects.push_back(dodeka.portals[i]);
		}
		
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0 && object->intersect(ray).t < 1) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return vec3(0.5f, 0.6f, 0.8f);
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 outRadiance(0,0,0);
		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				vec3 lightDir = light->position - hit.position;
				float l = length(lightDir);
				lightDir = normalize(lightDir);
				Ray shadowRay(hit.position + hit.normal * epsilon, lightDir);
				float cosTheta = dot(hit.normal, lightDir);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance / (l * l);
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightDir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = (outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess));
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);                           
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		if (hit.material->type == PORTAL) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float theta = (72.0f / 180.0f) * M_PI;
			vec3 reflect = reflectedDir * cosf(theta) + hit.normal * dot(reflectedDir, hit.normal) * (1 - cosf(theta)) + cross(hit.normal, reflectedDir) * sinf(theta);
			vec3 pos = hit.position - hit.center;
			vec3 rotatedPos = pos * cosf(theta) + hit.normal * dot(pos, hit.normal) * (1 - cosf(theta)) + cross(hit.normal, pos) * sinf(theta);
			hit.position = rotatedPos + hit.center;
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflect), depth + 1);
		}
		
		return outRadiance;
	}

	void moveCamera() {
		camera.Animate(0.1f);
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

std::vector<vec4> image(windowWidth* windowHeight);
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	scene.render(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case ' ': scene.moveCamera(); fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image); break;
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
long lastTime = 0;
void onIdle() {
	long t = glutGet(GLUT_ELAPSED_TIME);
	if (t - lastTime > 100) {
		lastTime = t;
		scene.moveCamera();
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	}
	glutPostRedisplay();
}