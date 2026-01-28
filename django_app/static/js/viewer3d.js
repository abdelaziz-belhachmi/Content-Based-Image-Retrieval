/**
 * Three.js 3D Model Viewer
 * 
 * A WebGL-based viewer for OBJ 3D models using Three.js
 */

class Model3DViewer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container #${containerId} not found`);
            return;
        }

        // Options with defaults
        this.options = {
            backgroundColor: options.backgroundColor || 0x1a1a2e,
            modelColor: options.modelColor || 0x00aaff,
            wireframe: options.wireframe || false,
            autoRotate: options.autoRotate !== undefined ? options.autoRotate : true,
            rotateSpeed: options.rotateSpeed || 0.5,
            ambientLightIntensity: options.ambientLightIntensity || 0.6,
            directionalLightIntensity: options.directionalLightIntensity || 0.8,
            ...options
        };

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.model = null;
        this.animationId = null;

        this.init();
    }

    init() {
        // Clear container
        this.container.innerHTML = '';

        // Get container dimensions
        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 400;

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);

        // Create camera
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(0, 0, 5);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Add OrbitControls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 50;
        this.controls.autoRotate = this.options.autoRotate;
        this.controls.autoRotateSpeed = this.options.rotateSpeed;

        // Add lights
        this.addLights();

        // Add grid helper
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x333333);
        this.scene.add(gridHelper);

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation loop
        this.animate();

        // Add loading placeholder
        this.showLoading();
    }

    addLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, this.options.ambientLightIntensity);
        this.scene.add(ambientLight);

        // Directional light 1
        const dirLight1 = new THREE.DirectionalLight(0xffffff, this.options.directionalLightIntensity);
        dirLight1.position.set(5, 10, 7.5);
        dirLight1.castShadow = true;
        this.scene.add(dirLight1);

        // Directional light 2 (back)
        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-5, 5, -5);
        this.scene.add(dirLight2);

        // Hemisphere light for better ambient
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.4);
        hemiLight.position.set(0, 20, 0);
        this.scene.add(hemiLight);
    }

    showLoading() {
        // Create loading overlay
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'viewer3d-loading';
        loadingDiv.innerHTML = `
            <div style="text-align: center; color: white;">
                <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                    <span class="visually-hidden">Chargement...</span>
                </div>
                <p class="mt-2">Chargement du modèle 3D...</p>
            </div>
        `;
        loadingDiv.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(26, 26, 46, 0.9);
            z-index: 10;
        `;
        this.container.style.position = 'relative';
        this.container.appendChild(loadingDiv);
    }

    hideLoading() {
        const loadingDiv = document.getElementById('viewer3d-loading');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    showError(message) {
        this.hideLoading();
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = `
            <div style="text-align: center; color: #dc3545;">
                <i class="bi bi-exclamation-triangle" style="font-size: 3rem;"></i>
                <p class="mt-2">${message}</p>
            </div>
        `;
        errorDiv.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(26, 26, 46, 0.9);
            z-index: 10;
        `;
        this.container.appendChild(errorDiv);
    }

    loadOBJ(url) {
        console.log('Loading OBJ from:', url);
        const loader = new THREE.OBJLoader();

        loader.load(
            url,
            (object) => {
                console.log('OBJ loaded successfully:', object);
                this.hideLoading();

                // Remove old model if exists
                if (this.model) {
                    this.scene.remove(this.model);
                }

                // Check if object has any children
                let hasMeshes = false;
                
                // Apply material to all meshes
                object.traverse((child) => {
                    if (child.isMesh) {
                        hasMeshes = true;
                        child.material = new THREE.MeshPhongMaterial({
                            color: this.options.modelColor,
                            wireframe: this.options.wireframe,
                            side: THREE.DoubleSide,
                            flatShading: false
                        });
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                
                if (!hasMeshes) {
                    console.warn('OBJ file loaded but contains no meshes');
                    this.showError('Le fichier OBJ ne contient pas de géométrie');
                    return;
                }

                // Center and scale the model
                const box = new THREE.Box3().setFromObject(object);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                
                console.log('Model bounding box:', { center, size });
                
                // Center the model
                object.position.sub(center);
                
                // Scale to fit in view
                const maxDim = Math.max(size.x, size.y, size.z);
                if (maxDim > 0) {
                    const scale = 2.5 / maxDim;
                    object.scale.setScalar(scale);
                }

                this.model = object;
                this.scene.add(this.model);

                // Reset camera position
                this.camera.position.set(0, 1.5, 4);
                this.controls.target.set(0, 0, 0);
                this.controls.update();

                // Dispatch load event
                this.container.dispatchEvent(new CustomEvent('modelLoaded', { 
                    detail: { 
                        vertices: this.countVertices(object),
                        faces: this.countFaces(object)
                    } 
                }));
            },
            (xhr) => {
                // Progress callback
                if (xhr.lengthComputable) {
                    const percentComplete = (xhr.loaded / xhr.total) * 100;
                    console.log(`Loading: ${Math.round(percentComplete)}%`);
                }
            },
            (error) => {
                console.error('Error loading OBJ:', error);
                console.error('URL was:', url);
                this.showError('Erreur lors du chargement du modèle 3D<br><small>' + url + '</small>');
            }
        );
    }

    countVertices(object) {
        let count = 0;
        object.traverse((child) => {
            if (child.isMesh && child.geometry) {
                const posAttr = child.geometry.getAttribute('position');
                if (posAttr) count += posAttr.count;
            }
        });
        return count;
    }

    countFaces(object) {
        let count = 0;
        object.traverse((child) => {
            if (child.isMesh && child.geometry) {
                if (child.geometry.index) {
                    count += child.geometry.index.count / 3;
                } else {
                    const posAttr = child.geometry.getAttribute('position');
                    if (posAttr) count += posAttr.count / 3;
                }
            }
        });
        return Math.floor(count);
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 400;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    // Public methods to control the viewer
    setWireframe(enabled) {
        this.options.wireframe = enabled;
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material.wireframe = enabled;
                }
            });
        }
    }

    setColor(color) {
        this.options.modelColor = color;
        if (this.model) {
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.material.color.setHex(color);
                }
            });
        }
    }

    setAutoRotate(enabled) {
        this.options.autoRotate = enabled;
        if (this.controls) {
            this.controls.autoRotate = enabled;
        }
    }

    resetView() {
        this.camera.position.set(0, 1.5, 4);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        if (this.controls) {
            this.controls.dispose();
        }

        // Remove event listeners
        window.removeEventListener('resize', this.onWindowResize);
        
        // Clear container
        this.container.innerHTML = '';
    }
}

// Export for use in other scripts
window.Model3DViewer = Model3DViewer;
