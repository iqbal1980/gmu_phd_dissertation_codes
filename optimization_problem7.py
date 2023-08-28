<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Object-Oriented Diagram</title>
    <style>
        canvas {
            border: 1px solid black;
        }
    </style>
</head>
<body>
    <button id="addRectangle">Add Rectangle</button>
    <canvas id="myCanvas" width="800" height="600"></canvas>
    <script>
        class Arrow {
            constructor(startX, startY, endX, endY) {
                this.startX = startX;
                this.startY = startY;
                this.endX = endX;
                this.endY = endY;
            }
            draw(ctx) {
                ctx.beginPath();
                ctx.moveTo(this.startX, this.startY);
                ctx.lineTo(this.endX, this.endY);
                ctx.stroke();
            }
        }
        class Rectangle {
            constructor(x, y, width, height) {
                this.x = x;
                this.y = y;
                this.width = width;
                this.height = height;
                this.isResizing = false;
                this.isDragging = false;
                this.selectedPointIndex = null;
                this.offsetX = 0;
                this.offsetY = 0;
            }
            draw(ctx) {
                ctx.fillStyle = "#00F";
                ctx.fillRect(this.x, this.y, this.width, this.height);
                const cornerPoints = [
                    { x: this.x, y: this.y },
                    { x: this.x + this.width, y: this.y },
                    { x: this.x, y: this.y + this.height },
                    { x: this.x + this.width, y: this.y + this.height }
                ];
                ctx.fillStyle = "#F00";
                cornerPoints.forEach(point => {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.closePath();
                });
                const middlePoints = [
                    { x: this.x + this.width / 2, y: this.y },
                    { x: this.x + this.width / 2, y: this.y + this.height },
                    { x: this.x, y: this.y + this.height / 2 },
                    { x: this.x + this.width, y: this.y + this.height / 2 }
                ];
                ctx.fillStyle = "#FF0";
                middlePoints.forEach(point => {
                    ctx.beginPath();
                    ctx.moveTo(point.x - 5, point.y);
                    ctx.lineTo(point.x, point.y - 5);
                    ctx.lineTo(point.x + 5, point.y);
                    ctx.lineTo(point.x, point.y + 5);
                    ctx.closePath();
                    ctx.fill();
                });
            }
        }
        class Diagram {
            constructor(ctx) {
                this.ctx = ctx;
                this.rectangles = [];
                this.arrows = [];
                this.isDrawingArrow = false;
                this.tempArrow = null;
                this.addEventListeners();
            }
            addRectangle(rectangle) {
                this.rectangles.push(rectangle);
                this.draw();
            }
            addArrow(arrow) {
                this.arrows.push(arrow);
                this.draw();
            }
            draw() {
                this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
                this.rectangles.forEach(rectangle => rectangle.draw(this.ctx));
                this.arrows.forEach(arrow => arrow.draw(this.ctx));
                if (this.isDrawingArrow && this.tempArrow) {
                    this.tempArrow.draw(this.ctx);
                }
            }
            addEventListeners() {
                this.ctx.canvas.addEventListener("mousedown", e => {
                    const rect = this.ctx.canvas.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;
                    this.rectangles.forEach(rectangle => {
                        const middlePoints = [
                            { x: rectangle.x + rectangle.width / 2, y: rectangle.y },
                            { x: rectangle.x + rectangle.width / 2, y: rectangle.y + rectangle.height },
                            { x: rectangle.x, y: rectangle.y + rectangle.height / 2 },
                            { x: rectangle.x + rectangle.width, y: rectangle.y + rectangle.height / 2 }
                        ];
                        for (let i = 0; i < middlePoints.length; i++) {
                            const dx = mouseX - middlePoints[i].x;
                            const dy = mouseY - middlePoints[i].y;
                            if (dx * dx + dy * dy <= 25) {
                                this.isDrawingArrow = true;
                                this.tempArrow = new Arrow(middlePoints[i].x, middlePoints[i].y, middlePoints[i].x, middlePoints[i].y);
                                return;
                            }
                        }
                    });
                    this.rectangles.forEach(rectangle => {
                        const cornerPoints = [
                            { x: rectangle.x, y: rectangle.y },
                            { x: rectangle.x + rectangle.width, y: rectangle.y },
                            { x: rectangle.x, y: rectangle.y + rectangle.height },
                            { x: rectangle.x + rectangle.width, y: rectangle.y + rectangle.height }
                        ];
                        for (let index = 0; index < cornerPoints.length; index++) {
                            const dx = mouseX - cornerPoints[index].x;
                            const dy = mouseY - cornerPoints[index].y;
                            if (dx * dx + dy * dy <= 25) {
                                rectangle.isResizing = true;
                                rectangle.selectedPointIndex = index;
                                return;
                            }
                        }
                        if (mouseX >= rectangle.x && mouseX <= rectangle.x + rectangle.width &&
                            mouseY >= rectangle.y && mouseY <= rectangle.y + rectangle.height) {
                            rectangle.isDragging = true;
                            rectangle.offsetX = mouseX - rectangle.x;
                            rectangle.offsetY = mouseY - rectangle.y;
                        }
                    });
                });
                this.ctx.canvas.addEventListener("mousemove", e => {
                    const rect = this.ctx.canvas.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;
                    if (this.isDrawingArrow && this.tempArrow) {
                        this.tempArrow.endX = mouseX;
                        this.tempArrow.endY = mouseY;
                        this.draw();
                    }
                    this.rectangles.forEach(rectangle => {
                        if (rectangle.isResizing) {
                            switch(rectangle.selectedPointIndex) {
                                case 0:
                                    rectangle.width += rectangle.x - mouseX;
                                    rectangle.height += rectangle.y - mouseY;
                                    rectangle.x = mouseX;
                                    rectangle.y = mouseY;
                                    break;
                                case 1:
                                    rectangle.width = mouseX - rectangle.x;
                                    rectangle.height += rectangle.y - mouseY;
                                    rectangle.y = mouseY;
                                    break;
                                case 2:
                                    rectangle.width += rectangle.x - mouseX;
                                    rectangle.height = mouseY - rectangle.y;
                                    rectangle.x = mouseX;
                                    break;
                                case 3:
                                    rectangle.width = mouseX - rectangle.x;
                                    rectangle.height = mouseY - rectangle.y;
                                    break;
                            }
                        }
                        if (rectangle.isDragging) {
                            rectangle.x = mouseX - rectangle.offsetX;
                            rectangle.y = mouseY - rectangle.offsetY;
                        }
                    });
                    this.draw();
                });
                this.ctx.canvas.addEventListener("mouseup", e => {
                    if (this.isDrawingArrow && this.tempArrow) {
                        this.addArrow(this.tempArrow);
                        this.isDrawingArrow = false;
                        this.tempArrow = null;
                    }
                    this.rectangles.forEach(rectangle => {
                        rectangle.isResizing = false;
                        rectangle.isDragging = false;
                        rectangle.selectedPointIndex = null;
                    });
                });
            }
        }
        const canvas = document.getElementById("myCanvas");
        const ctx = canvas.getContext("2d");
        const diagram = new Diagram(ctx);
        document.getElementById("addRectangle").addEventListener("click", () => {
            const newRectangle = new Rectangle(50, 50, 100, 50);
            diagram.addRectangle(newRectangle);
        });
    </script>
</body>
</html>
