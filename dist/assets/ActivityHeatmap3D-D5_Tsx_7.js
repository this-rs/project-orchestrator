import{r as B,j as n}from"./index-BC8bZ7vA.js";import{C as M,P as Be,V as _,W as ae,H as re,U as he,S as Y,a as N,b as oe,A as _e,M as Re,F as Fe,R as Pe,c as Ue,d as je,e as ze,f as Ee,g as Ne,E as Ae,h as ke,O as He,i as Le,j as Ve,D as Oe,k as Ie,l as Qe,m as W,G as Ge,B as Ye,n as me,o as de,p as We,q as $e,r as Xe,s as Ke,L as qe,t as Je,u as Ze}from"./RenderPass-BxNtBGba.js";const et={uniforms:{tDiffuse:{value:null},luminosityThreshold:{value:1},smoothWidth:{value:1},defaultColor:{value:new M(0)},defaultOpacity:{value:0}},vertexShader:`

		varying vec2 vUv;

		void main() {

			vUv = uv;

			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`

		uniform sampler2D tDiffuse;
		uniform vec3 defaultColor;
		uniform float defaultOpacity;
		uniform float luminosityThreshold;
		uniform float smoothWidth;

		varying vec2 vUv;

		void main() {

			vec4 texel = texture2D( tDiffuse, vUv );

			float v = luminance( texel.xyz );

			vec4 outputColor = vec4( defaultColor.rgb, defaultOpacity );

			float alpha = smoothstep( luminosityThreshold, luminosityThreshold + smoothWidth, v );

			gl_FragColor = mix( outputColor, texel, alpha );

		}`};class A extends Be{constructor(e,p=1,l,a){super(),this.strength=p,this.radius=l,this.threshold=a,this.resolution=e!==void 0?new _(e.x,e.y):new _(256,256),this.clearColor=new M(0,0,0),this.needsSwap=!1,this.renderTargetsHorizontal=[],this.renderTargetsVertical=[],this.nMips=5;let o=Math.round(this.resolution.x/2),m=Math.round(this.resolution.y/2);this.renderTargetBright=new ae(o,m,{type:re}),this.renderTargetBright.texture.name="UnrealBloomPass.bright",this.renderTargetBright.texture.generateMipmaps=!1;for(let T=0;T<this.nMips;T++){const F=new ae(o,m,{type:re});F.texture.name="UnrealBloomPass.h"+T,F.texture.generateMipmaps=!1,this.renderTargetsHorizontal.push(F);const U=new ae(o,m,{type:re});U.texture.name="UnrealBloomPass.v"+T,U.texture.generateMipmaps=!1,this.renderTargetsVertical.push(U),o=Math.round(o/2),m=Math.round(m/2)}const c=et;this.highPassUniforms=he.clone(c.uniforms),this.highPassUniforms.luminosityThreshold.value=a,this.highPassUniforms.smoothWidth.value=.01,this.materialHighPassFilter=new Y({uniforms:this.highPassUniforms,vertexShader:c.vertexShader,fragmentShader:c.fragmentShader}),this.separableBlurMaterials=[];const d=[6,10,14,18,22];o=Math.round(this.resolution.x/2),m=Math.round(this.resolution.y/2);for(let T=0;T<this.nMips;T++)this.separableBlurMaterials.push(this._getSeparableBlurMaterial(d[T])),this.separableBlurMaterials[T].uniforms.invSize.value=new _(1/o,1/m),o=Math.round(o/2),m=Math.round(m/2);this.compositeMaterial=this._getCompositeMaterial(this.nMips),this.compositeMaterial.uniforms.blurTexture1.value=this.renderTargetsVertical[0].texture,this.compositeMaterial.uniforms.blurTexture2.value=this.renderTargetsVertical[1].texture,this.compositeMaterial.uniforms.blurTexture3.value=this.renderTargetsVertical[2].texture,this.compositeMaterial.uniforms.blurTexture4.value=this.renderTargetsVertical[3].texture,this.compositeMaterial.uniforms.blurTexture5.value=this.renderTargetsVertical[4].texture,this.compositeMaterial.uniforms.bloomStrength.value=p,this.compositeMaterial.uniforms.bloomRadius.value=.1;const L=[1,.8,.6,.4,.2];this.compositeMaterial.uniforms.bloomFactors.value=L,this.bloomTintColors=[new N(1,1,1),new N(1,1,1),new N(1,1,1),new N(1,1,1),new N(1,1,1)],this.compositeMaterial.uniforms.bloomTintColors.value=this.bloomTintColors,this.copyUniforms=he.clone(oe.uniforms),this.blendMaterial=new Y({uniforms:this.copyUniforms,vertexShader:oe.vertexShader,fragmentShader:oe.fragmentShader,premultipliedAlpha:!0,blending:_e,depthTest:!1,depthWrite:!1,transparent:!0}),this._oldClearColor=new M,this._oldClearAlpha=1,this._basic=new Re,this._fsQuad=new Fe(null)}dispose(){for(let e=0;e<this.renderTargetsHorizontal.length;e++)this.renderTargetsHorizontal[e].dispose();for(let e=0;e<this.renderTargetsVertical.length;e++)this.renderTargetsVertical[e].dispose();this.renderTargetBright.dispose();for(let e=0;e<this.separableBlurMaterials.length;e++)this.separableBlurMaterials[e].dispose();this.compositeMaterial.dispose(),this.blendMaterial.dispose(),this._basic.dispose(),this._fsQuad.dispose()}setSize(e,p){let l=Math.round(e/2),a=Math.round(p/2);this.renderTargetBright.setSize(l,a);for(let o=0;o<this.nMips;o++)this.renderTargetsHorizontal[o].setSize(l,a),this.renderTargetsVertical[o].setSize(l,a),this.separableBlurMaterials[o].uniforms.invSize.value=new _(1/l,1/a),l=Math.round(l/2),a=Math.round(a/2)}render(e,p,l,a,o){e.getClearColor(this._oldClearColor),this._oldClearAlpha=e.getClearAlpha();const m=e.autoClear;e.autoClear=!1,e.setClearColor(this.clearColor,0),o&&e.state.buffers.stencil.setTest(!1),this.renderToScreen&&(this._fsQuad.material=this._basic,this._basic.map=l.texture,e.setRenderTarget(null),e.clear(),this._fsQuad.render(e)),this.highPassUniforms.tDiffuse.value=l.texture,this.highPassUniforms.luminosityThreshold.value=this.threshold,this._fsQuad.material=this.materialHighPassFilter,e.setRenderTarget(this.renderTargetBright),e.clear(),this._fsQuad.render(e);let c=this.renderTargetBright;for(let d=0;d<this.nMips;d++)this._fsQuad.material=this.separableBlurMaterials[d],this.separableBlurMaterials[d].uniforms.colorTexture.value=c.texture,this.separableBlurMaterials[d].uniforms.direction.value=A.BlurDirectionX,e.setRenderTarget(this.renderTargetsHorizontal[d]),e.clear(),this._fsQuad.render(e),this.separableBlurMaterials[d].uniforms.colorTexture.value=this.renderTargetsHorizontal[d].texture,this.separableBlurMaterials[d].uniforms.direction.value=A.BlurDirectionY,e.setRenderTarget(this.renderTargetsVertical[d]),e.clear(),this._fsQuad.render(e),c=this.renderTargetsVertical[d];this._fsQuad.material=this.compositeMaterial,this.compositeMaterial.uniforms.bloomStrength.value=this.strength,this.compositeMaterial.uniforms.bloomRadius.value=this.radius,this.compositeMaterial.uniforms.bloomTintColors.value=this.bloomTintColors,e.setRenderTarget(this.renderTargetsHorizontal[0]),e.clear(),this._fsQuad.render(e),this._fsQuad.material=this.blendMaterial,this.copyUniforms.tDiffuse.value=this.renderTargetsHorizontal[0].texture,o&&e.state.buffers.stencil.setTest(!0),this.renderToScreen?(e.setRenderTarget(null),this._fsQuad.render(e)):(e.setRenderTarget(l),this._fsQuad.render(e)),e.setClearColor(this._oldClearColor,this._oldClearAlpha),e.autoClear=m}_getSeparableBlurMaterial(e){const p=[],l=e/3;for(let a=0;a<e;a++)p.push(.39894*Math.exp(-.5*a*a/(l*l))/l);return new Y({defines:{KERNEL_RADIUS:e},uniforms:{colorTexture:{value:null},invSize:{value:new _(.5,.5)},direction:{value:new _(.5,.5)},gaussianCoefficients:{value:p}},vertexShader:`

				varying vec2 vUv;

				void main() {

					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

				}`,fragmentShader:`

				#include <common>

				varying vec2 vUv;

				uniform sampler2D colorTexture;
				uniform vec2 invSize;
				uniform vec2 direction;
				uniform float gaussianCoefficients[KERNEL_RADIUS];

				void main() {

					float weightSum = gaussianCoefficients[0];
					vec3 diffuseSum = texture2D( colorTexture, vUv ).rgb * weightSum;

					for ( int i = 1; i < KERNEL_RADIUS; i ++ ) {

						float x = float( i );
						float w = gaussianCoefficients[i];
						vec2 uvOffset = direction * invSize * x;
						vec3 sample1 = texture2D( colorTexture, vUv + uvOffset ).rgb;
						vec3 sample2 = texture2D( colorTexture, vUv - uvOffset ).rgb;
						diffuseSum += ( sample1 + sample2 ) * w;

					}

					gl_FragColor = vec4( diffuseSum, 1.0 );

				}`})}_getCompositeMaterial(e){return new Y({defines:{NUM_MIPS:e},uniforms:{blurTexture1:{value:null},blurTexture2:{value:null},blurTexture3:{value:null},blurTexture4:{value:null},blurTexture5:{value:null},bloomStrength:{value:1},bloomFactors:{value:null},bloomTintColors:{value:null},bloomRadius:{value:0}},vertexShader:`

				varying vec2 vUv;

				void main() {

					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

				}`,fragmentShader:`

				varying vec2 vUv;

				uniform sampler2D blurTexture1;
				uniform sampler2D blurTexture2;
				uniform sampler2D blurTexture3;
				uniform sampler2D blurTexture4;
				uniform sampler2D blurTexture5;
				uniform float bloomStrength;
				uniform float bloomRadius;
				uniform float bloomFactors[NUM_MIPS];
				uniform vec3 bloomTintColors[NUM_MIPS];

				float lerpBloomFactor( const in float factor ) {

					float mirrorFactor = 1.2 - factor;
					return mix( factor, mirrorFactor, bloomRadius );

				}

				void main() {

					// 3.0 for backwards compatibility with previous alpha-based intensity
					vec3 bloom = 3.0 * bloomStrength * (
						lerpBloomFactor( bloomFactors[ 0 ] ) * bloomTintColors[ 0 ] * texture2D( blurTexture1, vUv ).rgb +
						lerpBloomFactor( bloomFactors[ 1 ] ) * bloomTintColors[ 1 ] * texture2D( blurTexture2, vUv ).rgb +
						lerpBloomFactor( bloomFactors[ 2 ] ) * bloomTintColors[ 2 ] * texture2D( blurTexture3, vUv ).rgb +
						lerpBloomFactor( bloomFactors[ 3 ] ) * bloomTintColors[ 3 ] * texture2D( blurTexture4, vUv ).rgb +
						lerpBloomFactor( bloomFactors[ 4 ] ) * bloomTintColors[ 4 ] * texture2D( blurTexture5, vUv ).rgb
					);

					float bloomAlpha = max( bloom.r, max( bloom.g, bloom.b ) );
					gl_FragColor = vec4( bloom, bloomAlpha );

				}`})}}A.BlurDirectionX=new _(1,0);A.BlurDirectionY=new _(0,1);const fe={note_created:"#3B82F6",note_confirmed:"#4ade80",decision:"#8B5CF6",commit:"#64748B",skill_created:"#EC4899",skill_activated:"#fbbf24",protocol_transition:"#F97316"},pe=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],ge=new M(2282478);function at({events:k,projectColorMap:e}){const p=B.useRef(null),l=B.useRef(0),a=B.useRef(null),o=B.useRef(new Pe),m=B.useRef(new _(-999,-999)),[c,d]=B.useState(null),{grid:L,perProjectGrid:T,maxCount:F,eventsByCell:U,dayTotals:$,sortedSlugs:X}=B.useMemo(()=>{const r=Array.from({length:7},()=>Array(24).fill(0)),u=Array.from({length:7},()=>Array.from({length:24},()=>new Map)),h=new Map;for(const f of k){const D=f.date.getDay(),v=f.date.getHours();r[D][v]++;const V=`${D}-${v}`,O=h.get(V)||[];if(O.push(f),h.set(V,O),e){const S=f.projectSlug||"_global",I=u[D][v];I.set(S,(I.get(S)||0)+1)}}let i=0;for(const f of r)for(const D of f)D>i&&(i=D);const b=r.map(f=>f.reduce((D,v)=>D+v,0)),g=e?Array.from(new Set(k.map(f=>f.projectSlug||"_global"))).sort():[];return{grid:r,perProjectGrid:u,maxCount:i||1,eventsByCell:h,dayTotals:b,sortedSlugs:g}},[k,e]),K=B.useCallback(r=>{if(!p.current)return;const u=p.current.getBoundingClientRect();m.current.x=(r.clientX-u.left)/u.width*2-1,m.current.y=-((r.clientY-u.top)/u.height)*2+1},[]),q=B.useCallback(()=>{if(m.current.set(-999,-999),d(null),a.current){const r=a.current.userData,u=a.current.material;u.emissiveIntensity=r.baseEmissive,a.current=null}},[]);return B.useEffect(()=>{const r=p.current;if(!r)return;const u=r.clientWidth,h=360,i=new Ue;i.background=new M(527380);const b=new je(45,u/h,.1,200);b.position.set(20,22,20),b.lookAt(12,0,3.5);const g=new ze({antialias:!0,alpha:!1});g.setSize(u,h),g.setPixelRatio(Math.min(window.devicePixelRatio,2)),g.shadowMap.enabled=!0,g.shadowMap.type=Ee,g.toneMapping=Ne,g.toneMappingExposure=1.4,r.appendChild(g.domElement);const f=new Ae(g);f.addPass(new ke(i,b));const D=new A(new _(u,h),.4,.6,.9);f.addPass(D);const v=new He(b,g.domElement);v.enableDamping=!0,v.dampingFactor=.05,v.target.set(12,0,3.5),v.minDistance=10,v.maxDistance=60,v.maxPolarAngle=Math.PI/2.1,v.update();const V=new Le(6583435,1.5);i.add(V);const O=new Ve(2282478,1976635,.6);i.add(O);const S=new Oe(13358561,.5);S.position.set(-10,15,10),S.castShadow=!0,S.shadow.mapSize.set(1024,1024),S.shadow.camera.left=-15,S.shadow.camera.right=30,S.shadow.camera.top=12,S.shadow.camera.bottom=-5,i.add(S);const I=new Ie(40,20),xe=new Qe({color:791074,roughness:.95,metalness:.05}),Q=new W(I,xe);Q.rotation.x=-Math.PI/2,Q.position.set(12,-.01,3.5),Q.receiveShadow=!0,i.add(Q);const ie=new Ge(40,40,1450034,988964);ie.position.set(12,0,3.5),i.add(ie);const J=[],Z=new Ye(.85,1,.85);Z.translate(0,.5,0);const be=!!e&&X.length>1;for(let s=0;s<7;s++)for(let t=0;t<24;t++){const w=L[s][t],x=w/F,y=w===0?.12:.25+x*7,R=w===0;if(be&&!R){const j=T[s][t];let H=0;for(const z of X){const E=j.get(z)||0;if(E===0)continue;const te=E/w,C=y*te,G=E/F,Te=e.get(z)||"#64748B",se=new M(Te),Me=se.clone().multiplyScalar(.4),ce=.5+G*.45,Se=Math.max(.05,.45-G*.4),ue=.15+G*.4,Ce=new M().lerpColors(Me,se,Math.max(.4,G)),De=new me({color:Ce,emissive:se.clone(),emissiveIntensity:ue,transparent:!0,opacity:ce,transmission:Se,roughness:.15,metalness:0,thickness:.8,ior:1.4,envMapIntensity:.5,side:de}),P=new W(Z,De);P.position.set(t,H,s),P.scale.y=C,P.castShadow=!0,P.receiveShadow=!0,P.userData={day:s,hour:t,count:w,segCount:E,projectSlug:z,intensity:x,baseEmissive:ue,baseScaleY:C,baseOpacity:ce,baseY:H},i.add(P),J.push(P),H+=C}}else{const j=R?.25:.4+x*.5,H=R?.8:Math.max(.05,.55-x*.5),z=R?.05:.1+x*.45,E=new M().lerpColors(new M(1400437),ge,R?.15:Math.max(.3,x)),te=new me({color:E,emissive:ge.clone(),emissiveIntensity:z,transparent:!0,opacity:j,transmission:H,roughness:.15,metalness:0,thickness:.8,ior:1.4,envMapIntensity:.5,side:de}),C=new W(Z,te);C.position.set(t,0,s),C.scale.y=y,C.castShadow=!R,C.receiveShadow=!0,C.userData={day:s,hour:t,count:w,intensity:x,baseEmissive:z,baseScaleY:y,baseOpacity:j},i.add(C),J.push(C)}}const we=Math.max(...$,1),ee=$.map((s,t)=>({day:t,total:s,t:s/we})).filter(s=>s.total>0).sort((s,t)=>t.total-s.total).slice(0,3);for(const{day:s,t}of ee){if(t<.4)continue;const w=1.5+t*3,x=new M().lerpColors(new M(959977),new M(14742270),t*.6),y=new We(x,w,30,Math.PI/5,.6,1.2);y.position.set(11.5,14,s),y.target.position.set(11.5,0,s),y.castShadow=!0,y.shadow.mapSize.set(512,512),i.add(y),i.add(y.target)}if(ee.length===0||ee[0].t<.4){const s=new $e(2282478,1,50);s.position.set(12,14,3.5),i.add(s)}for(let s=0;s<7;s++){const t=ve(pe[s],{fontSize:28,color:"#64748b"});t.position.set(-1.5,.1,s),t.scale.set(1.6,.5,1),i.add(t)}for(let s=0;s<24;s+=3){const t=ve(`${s}h`,{fontSize:24,color:"#475569"});t.position.set(s,.1,7.5),t.scale.set(1.2,.4,1),i.add(t)}r.addEventListener("mousemove",K),r.addEventListener("mouseleave",q);const ye=()=>{if(!r)return;const s=r.clientWidth;b.aspect=s/h,b.updateProjectionMatrix(),g.setSize(s,h),f.setSize(s,h)},ne=new ResizeObserver(ye);ne.observe(r);const le=()=>{l.current=requestAnimationFrame(le),v.update(),o.current.setFromCamera(m.current,b);const s=o.current.intersectObjects(J);if(a.current){const t=a.current.userData,w=a.current.material;w.emissiveIntensity=t.baseEmissive,w.opacity=t.baseOpacity,a.current.scale.y=t.baseScaleY,a.current=null}if(s.length>0){const t=s[0].object;if(t.userData.count>0){a.current=t;const w=t.material;w.emissiveIntensity=Math.min(t.userData.baseEmissive+.35,.9),w.opacity=Math.min(t.userData.baseOpacity+.2,1),t.scale.y=t.userData.baseScaleY*1.06;const x=new N;x.copy(t.position),x.y=(t.userData.baseY??0)+t.scale.y+.5,x.project(b);const y=r.getBoundingClientRect(),R=(x.x+1)/2*y.width,j=(-x.y+1)/2*y.height;d({x:R,y:j,day:pe[t.userData.day],hour:t.userData.hour,count:t.userData.count,events:U.get(`${t.userData.day}-${t.userData.hour}`)||[]})}else d(null)}else d(null);f.render()};return le(),()=>{cancelAnimationFrame(l.current),r.removeEventListener("mousemove",K),r.removeEventListener("mouseleave",q),ne.disconnect(),v.dispose(),g.dispose(),f.dispose(),i.traverse(s=>{s instanceof W&&(s.geometry.dispose(),s.material instanceof Xe&&s.material.dispose())}),r.contains(g.domElement)&&r.removeChild(g.domElement)}},[L,T,F,U,$,X,e,K,q]),n.jsxs("div",{className:"relative",children:[n.jsx("div",{ref:p,className:"w-full rounded-lg overflow-hidden",style:{height:360,cursor:"grab"}}),c&&n.jsx("div",{className:"absolute z-50 pointer-events-none",style:{left:c.x,top:c.y,transform:"translate(-50%, -110%)"},children:n.jsxs("div",{className:"bg-slate-900/95 backdrop-blur-sm border border-slate-700/80 rounded-lg px-3 py-2 shadow-xl min-w-[180px] max-w-[260px]",children:[n.jsxs("div",{className:"flex items-center justify-between mb-1.5",children:[n.jsxs("span",{className:"text-[10px] font-medium text-cyan-400",children:[c.day," ",c.hour,":00–",c.hour+1,":00"]}),n.jsxs("span",{className:"text-[10px] text-slate-500 font-mono",children:[c.count," event",c.count!==1?"s":""]})]}),n.jsx("div",{className:"space-y-1",children:e?(()=>{const r=new Map;for(const u of c.events){const h=u.projectName||u.projectSlug||"Global",i=r.get(h)||[];i.push(u),r.set(h,i)}return Array.from(r.entries()).map(([u,h])=>{var i;return n.jsxs("div",{className:"mb-1 last:mb-0",children:[n.jsxs("div",{className:"flex items-center gap-1.5 mb-0.5",children:[n.jsx("div",{className:"w-2 h-2 rounded-full shrink-0",style:{backgroundColor:(i=h[0])!=null&&i.projectSlug?e.get(h[0].projectSlug)||"#6B7280":"#64748B"}}),n.jsx("span",{className:"text-[9px] font-medium text-slate-300",children:u}),n.jsxs("span",{className:"text-[8px] text-slate-600",children:["(",h.length,")"]})]}),h.slice(0,2).map(b=>n.jsxs("div",{className:"flex items-center gap-1.5 ml-3.5",children:[n.jsx("div",{className:"w-1 h-1 rounded-full shrink-0",style:{backgroundColor:fe[b.type]}}),n.jsx("span",{className:"text-[8px] text-slate-500 truncate",children:b.label})]},b.id)),h.length>2&&n.jsxs("span",{className:"text-[7px] text-slate-600 ml-3.5",children:["+",h.length-2," more"]})]},u)})})():n.jsxs(n.Fragment,{children:[c.events.slice(0,4).map(r=>n.jsxs("div",{className:"flex items-center gap-1.5",children:[n.jsx("div",{className:"w-1.5 h-1.5 rounded-full shrink-0",style:{backgroundColor:fe[r.type]}}),n.jsx("span",{className:"text-[9px] text-slate-400 truncate",children:r.label})]},r.id)),c.events.length>4&&n.jsxs("span",{className:"text-[8px] text-slate-600",children:["+",c.events.length-4," more"]})]})})]})}),n.jsx("div",{className:"absolute bottom-2 right-3 text-[9px] text-slate-600 pointer-events-none",children:"Drag to rotate · Scroll to zoom"})]})}function ve(k,e={}){const p=e.fontSize??32,l=e.color??"#94a3b8",a=document.createElement("canvas");a.width=128,a.height=48;const o=a.getContext("2d");o.clearRect(0,0,a.width,a.height),o.font=`${p}px -apple-system, BlinkMacSystemFont, sans-serif`,o.textAlign="center",o.textBaseline="middle",o.fillStyle=l,o.fillText(k,a.width/2,a.height/2);const m=new Ke(a);m.minFilter=qe;const c=new Je({map:m,transparent:!0});return new Ze(c)}export{at as ActivityHeatmap3D};
